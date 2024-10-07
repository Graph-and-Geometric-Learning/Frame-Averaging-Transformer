import os
import wandb
import random
import pickle
import logging
import argparse

import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import auc, precision_recall_curve

from faformer.model.predictor import AptamerScreener
from faformer.data.protein_complex_dataset import ContactPredDataset, AptamerDataset
from faformer.data.protein_complex_dataloader import ContactPredDataLoader, AptamerDataLoader
from faformer.utils.utils import set_random_seed

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def evaluate_affinity(preds, labels, top_k=[1, 5, 10]): 
    # scores: [N], labels: [N]
    top_k_precisions = []
    top_k_recalls = []
    for k in top_k:
        topk_preds = torch.topk(preds, k=k, dim=-1).indices
        top_k_precisions.append(labels[topk_preds].sum(dim=-1) / k)
        top_k_recalls.append(labels[topk_preds].sum(dim=-1) / labels.sum())
    top_k_precisions = [p.item() for p in top_k_precisions]
    top_k_recalls = [r.item() for r in top_k_recalls]
    
    precision_curve, recall_curve, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall_curve, precision_curve)

    return top_k_precisions, top_k_recalls, pr_auc


def batch_evaluate_contact_map(pred_binding_site, y_contact, mask, pos_weight=1.0):
    # mask: [B, -1]
    # return: [B], a list of metrics for each sample
    # positive precision
    precision = ((pred_binding_site > 0).float() * y_contact).float()
    precision = (precision * mask).sum(dim=-1) / (((pred_binding_site > 0) * mask).sum(dim=-1) + 1e-8)
    # positive recall
    recall = ((pred_binding_site > 0).float() * y_contact).float()
    recall = (recall * mask).sum(dim=-1) / (y_contact.float().sum(dim=-1) + 1e-8)
    # positive F1
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # PR-AUC
    pr_auc = []
    for i in range(pred_binding_site.shape[0]):
        y_true = y_contact[i][mask[i] > 0].detach().cpu().numpy()
        y_score = pred_binding_site[i][mask[i] > 0].detach().cpu().numpy()
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc.append(auc(recall_curve, precision_curve))
    # loss
    loss = F.binary_cross_entropy_with_logits(pred_binding_site, y_contact, reduction="none", pos_weight=torch.tensor(pos_weight).cuda())
    loss = (loss * mask).sum() / mask.sum()
    return loss, precision.cpu().tolist(), recall.cpu().tolist(), f1.cpu().tolist(), pr_auc


def train(epoch, model, device, loader, optimizer, clip_norm, pos_weight):
    model.train()

    pos_acc_list, pos_recall_list, pos_f1_list, pr_auc_list = [], [], [], []
    total_loss = 0.
    epoch_iter = tqdm(loader, ncols=130)
    for step, batch in enumerate(epoch_iter):
        partner_batch, prot_batch, contact_map = batch

        contact_map = contact_map.to(device)
        partner_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in partner_batch]
        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]
        
        pred_map, true_pred_map, mask = model(partner_batch, prot_batch, contact_map)
        loss, pos_acc, pos_recall, pos_f1, pr_auc = batch_evaluate_contact_map(pred_map, true_pred_map, mask, pos_weight=pos_weight)

        pos_acc_list += pos_acc
        pos_recall_list += pos_recall
        pos_f1_list += pos_f1
        pr_auc_list += pr_auc

        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.cpu().item()
        epoch_iter.set_description(f"epoch: {epoch}, train_loss: {loss.cpu().item():.4f}")

    pos_acc, pos_recall, pos_f1, pr_auc = np.mean(pos_acc_list), np.mean(pos_recall_list), np.mean(pos_f1_list), np.mean(pr_auc_list)
    return {"loss": total_loss / (step + 1), "precision": pos_acc, "recall": pos_recall, "f1": pos_f1, "pr_auc": pr_auc}


@torch.no_grad()
def test(model, device, loader, threshold, topk_list=[10, 20, 50], largest=False):
    model.eval()

    gt_affinity = []
    pred_affinity = []
    protein_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in loader.get_protein()]
    prot_embs = model.inference_embs(*protein_batch)

    for step, batch in enumerate(loader):
        aptamer_batch, affinity_scores = batch

        aptamer_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in aptamer_batch]
        pred_scores = model.zero_shot_affinity(prot_embs, aptamer_batch)
        
        gt_affinity.append(affinity_scores)
        pred_affinity.append(pred_scores)

    gt_affinity = torch.cat(gt_affinity, dim=0)
    pred_affinity = torch.cat(pred_affinity, dim=0).squeeze()
    if largest:
        labels = (gt_affinity > threshold).float()
    else:
        labels = (gt_affinity < threshold).float()
    n_pos = int(labels.sum().cpu().item())

    top_k_precisions, top_k_recalls, prauc = evaluate_affinity(pred_affinity.cpu(), labels.cpu(), top_k=topk_list+[n_pos])
    return {"precision": top_k_precisions, "recall":top_k_recalls, "prauc": prauc}


def read_data(data_dict, batch_size, dataset_name, protein_emb_path, rna_emb_path, cache_path, device=0, aptamer_prot=None, train=True):
    if train:
        dataset = ContactPredDataset(dataset_name, data_dict, cache_path, protein_emb_path, rna_emb_path, save_hetero=False, device=device)
        return ContactPredDataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = AptamerDataset(aptamer_prot, data_dict, cache_path, protein_emb_path, rna_emb_path, save_hetero=True, device=device)
        return AptamerDataLoader(dataset, batch_size=batch_size, shuffle=False)


def run(seed, args):
    save_path = args.save_path
    device = args.device

    s = time()
    train_dict = pickle.load(open(args.data_path["train"], 'rb'))
    train_loader = read_data(train_dict, args.batch_size, args.complex_type, args.train_esm_path, args.train_fm_path, None, device=device)
    protein, val_dict, test_dict = pickle.load(open(args.data_path["eval"], 'rb'))
    val_loader = read_data(val_dict, args.eval_batch_size, args.complex_type, args.eval_esm_path, args.val_fm_path, args.cache_path["val"], device=device, aptamer_prot=protein, train=False)
    test_loader = read_data(test_dict, args.eval_batch_size, args.complex_type, args.eval_esm_path, args.test_fm_path, args.cache_path["test"], device=device, aptamer_prot=protein, train=False)
    logging.info(f"Data loading time: {time() - s:.4f}s")

    model = AptamerScreener(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.save_checkpoint and not os.path.exists(save_path):
        os.makedirs(save_path)

    train_curve, val_curve, test_curve = [], [], []
    best_val_epoch, best_val_metric, best_model_weight = 0, 0., model.state_dict()

    for epoch in range(1, args.epochs + 1):
        train_perf = train(epoch, model, device, train_loader, optimizer, args.clip_norm, args.pos_weight)
        train_curve.append(train_perf["loss"])

        if epoch % args.eval_step == 0:
            val_perf = test(model, device, val_loader, args.pos_threshold, args.topk_list, args.largest)
            test_perf = test(model, device, test_loader, args.pos_threshold, args.topk_list, args.largest)

            print({f'Train':train_perf})
            print({f"Val":val_perf})
            print({f"Test":test_perf})

            if args.use_wandb:
                wandb.log({f"Train_{k}":v for k,v in train_perf.items()})
                wandb.log({f"Val_{k}":v for k,v in val_perf.items()})
                wandb.log({f"Test_{k}":v for k,v in test_perf.items()})

            eval_metric = val_perf["precision"][2]
            if eval_metric > best_val_metric:
                best_val_epoch = epoch
                best_val_metric = eval_metric
                best_model_weight = deepcopy(model.state_dict())
            
            print(f"Best val score: {best_val_metric}, best epoch: {best_val_epoch}")
            val_curve.append(val_perf)
            test_curve.append(test_perf)
        else:
            print({f'Train': train_perf})

    if args.save_checkpoint:
        torch.save({"model": best_model_weight}, f"{save_path}/{seed}_best.pt")

    print('Best val score: {}'.format(val_curve[best_val_epoch - 1]))
    print('Best test score: {}'.format(test_curve[best_val_epoch - 1]))

    if args.use_wandb:
        wandb.log({f'Best test score, seed {seed}': test_curve[best_val_epoch - 1]})

    logging.info(f"Overall time: {time() - s:.4f}s")
    return val_curve[best_val_epoch - 1], test_curve[best_val_epoch - 1]


def main(args):
    checkpoint_folder = "AptamerScreen" \
                    + "-" + str(args.model) \
                    + "-" + str(args.lr) \
                    + "-" + str(args.batch_size) \
                    + "-" + str(args.clip_norm) \
                    + "-" + str(args.n_layers) \
                    + "-" + str(args.hidden_dim) \
                    + "-" + str(args.top_k_neighbors) \
                    + "-" + str(args.drop_ratio) \
                    + "-" + str(args.attn_drop_ratio) \
                    + "-" + str(args.pos_weight) \
                    + "-" + str(args.max_dist) \

    args.save_path = os.path.join(args.save_dir, checkpoint_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    args.complex_type = "prot_rna"

    args.train_esm_path = os.path.join(args.dataset_root, f"embeddings/aptamer/train_esm_dict.pkl")
    args.eval_esm_path = os.path.join(args.dataset_root, f"embeddings/aptamer/{args.dataset}_esm_dict.pkl")

    args.train_fm_path = os.path.join(args.dataset_root, f"embeddings/aptamer/train_fm_dict.pkl")
    args.val_fm_path = os.path.join(args.dataset_root, f"embeddings/aptamer/{args.dataset}_val_fm_dict.pkl")
    args.test_fm_path = os.path.join(args.dataset_root, f"embeddings/aptamer/{args.dataset}_test_fm_dict.pkl")

    args.data_path = {
        "train": os.path.join(args.dataset_root, f"aptamer/train.pkl"),
        "eval": os.path.join(args.dataset_root, f"aptamer/{args.dataset}_eval.pkl"),
    }
    args.cache_path = {"val": os.path.join(args.dataset_root, f"cache/{args.dataset}_val_hetero.pkl"),
                       "test": os.path.join(args.dataset_root, f"cache/{args.dataset}_test_hetero.pkl"),}

    threshold = {
                "GFP": 10, "NELF": 5, "CHK2": 100, "UBLCP1": 200, "HNRNPC": -0.5
            }  # binding classification threshold
    args.pos_threshold = threshold[args.dataset]

    if args.dataset in ["GFP", "NELF", "HNRNPC"]:
        args.largest = False
    else:
        args.largest = True
    
    args.topk_list = [10, 20, 50]

    """record"""
    if args.use_wandb:
        exp_name = checkpoint_folder 
        wandb.init(project="contact_prediction", name=exp_name)
        wandb.config.update(args)
        # wandb.watch(model, log='all')

    logging.info(f"Save path: {args.save_path}")
    print(args)

    test_metric_list = []
    for seed in args.seeds:
        args.seed = seed
        set_random_seed(args.seed)
        val_metric, test_metric = run(args.seed, args)
        test_metric_list.append(test_metric)

    mean_test_metric_list = [[]]
    std_test_metric_list = [[]]
    for k in range(len(test_metric_list[0]["precision"])):
        mean_test_metric_list[0].append(np.mean([test_metric["precision"][k] for test_metric in test_metric_list]))
        std_test_metric_list[0].append(np.std([test_metric["precision"][k] for test_metric in test_metric_list]))

    mean_test_metric_list.append(np.mean([test_metric["prauc"] for test_metric in test_metric_list]))
    std_test_metric_list.append(np.std([test_metric["prauc"] for test_metric in test_metric_list]))
    print(mean_test_metric_list)
    print(std_test_metric_list)

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs="+", default=[0,1,2])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--save_checkpoint", action="store_true", default=False)
    parser.add_argument('--dataset_root', type=str, default="/home/tinglin/Frame-Averaging-Transformer/dataset/")
    parser.add_argument('--save_dir', type=str, default="/home/tinglin/Frame-Averaging-Transformer/dataset/checkpoints/")
    parser.add_argument('--dataset', type=str, default='GFP', help="GFP,HNRNPC,NELF,CHK2,UBLCP1")

    """training parameter"""
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--pos_weight', type=float, default=4.)
    parser.add_argument('--threshold', type=float, default=6.0)

    """model parameter"""
    parser.add_argument('--model', type=str, default='faformer')
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--edge_hidden_dim', type=int, default=64)
    parser.add_argument('--drop_ratio', type=float, default=0.2)
    parser.add_argument('--attn_drop_ratio', type=float, default=0.2)
    parser.add_argument('--top_k_neighbors', type=int, default=30)
    parser.add_argument('--embedding_grad_frac', type=float, default=1)
    parser.add_argument('--max_dist', type=float, default=1e5)
    parser.add_argument('--act', type=str, default='swiglu')
    parser.add_argument("--edge_residue", default=True)

    args = parser.parse_args()
    main(args)
