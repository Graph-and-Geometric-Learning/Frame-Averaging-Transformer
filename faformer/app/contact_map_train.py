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

from faformer.model.predictor import ContactMapPredictor
from faformer.data.protein_complex_dataset import ContactPredDataset
from faformer.data.protein_complex_dataloader import ContactPredDataLoader
from utils.utils import (
    set_random_seed, 
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


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
        rna_batch, prot_batch, contact_map = batch

        contact_map = contact_map.to(device)
        rna_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in rna_batch]
        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]
        
        pred_map, true_pred_map, mask = model(rna_batch, prot_batch, contact_map)
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
def test(model, device, loader, pos_weight):
    model.eval()
    loss_list, precision_list, recall_list, f1_list, pr_auc_list = [], [], [], [], []
    pr_auc_list = []

    for step, batch in enumerate(loader):
        rna_batch, prot_batch, contact_map = batch

        contact_map = contact_map.to(device)
        rna_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in rna_batch]
        prot_batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in prot_batch]

        pred_map, true_pred_map, mask = model(rna_batch, prot_batch, contact_map)
        loss, precision, recall, f1, pr_auc = batch_evaluate_contact_map(pred_map, true_pred_map, mask, pos_weight=pos_weight)

        loss_list.append(loss.cpu().item())
        precision_list += precision
        recall_list += recall
        f1_list += f1
        pr_auc_list += pr_auc

    loss, precision, recall, f1, pr_auc = np.mean(loss_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list), np.mean(pr_auc_list)
    return {"loss": loss, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


def read_data(data_dict, batch_size, dataset_name, protein_emb_path, rna_emb_path, cache_path, device=0, shuffle=True):
    dataset = ContactPredDataset(dataset_name, data_dict, cache_path, protein_emb_path, rna_emb_path, save_hetero=True, device=device)
    return ContactPredDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run(seed, args):
    save_path = args.save_path
    device = args.device

    s = time()
    train_dict, val_dict, test_dict = pickle.load(open(args.data_path, 'rb'))
    train_loader = read_data(train_dict, args.batch_size, args.complex_type, args.train_esm_path, args.train_fm_path, args.cache_path["train"], device=device, shuffle=True)
    val_loader = read_data(val_dict, args.eval_batch_size, args.complex_type, args.val_esm_path, args.val_fm_path, args.cache_path["val"], device=device, shuffle=False)
    test_loader = read_data(test_dict, args.eval_batch_size, args.complex_type, args.test_esm_path, args.test_fm_path, args.cache_path["test"], device=device, shuffle=False)    
    logging.info(f"Data loading time: {time() - s:.4f}s")

    model = ContactMapPredictor(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_curve, val_curve, test_curve = [], [], []
    if args.save_checkpoint and not os.path.exists(save_path):
        os.makedirs(save_path)

    best_val_epoch, best_val_metric = 0, 0.
    for epoch in range(1, args.epochs + 1):
        train_perf = train(epoch, model, device, train_loader, optimizer, args.clip_norm, args.pos_weight)
        train_curve.append(train_perf["loss"])

        if epoch % args.eval_step == 0:
            val_perf = test(model, device, val_loader, args.pos_weight)
            test_perf = test(model, device, test_loader, args.pos_weight)

            print({f'Train':train_perf})
            print({f"Val":val_perf})
            print({f"Test":test_perf})

            if args.use_wandb:
                wandb.log({f"Train_{k}":v for k,v in train_perf.items()} | {f"Val_{k}":v for k,v in val_perf.items()} | {f"Test_{k}":v for k,v in test_perf.items()})

            if val_perf["f1"] > best_val_metric:
                best_val_epoch = epoch
                best_val_metric = val_perf["f1"]
            
            print(f"Best val score: {best_val_metric}, best epoch: {best_val_epoch}")
            
            val_curve.append(val_perf)
            test_curve.append(test_perf)
        else:
            print({f'Train': train_perf})

    if args.save_checkpoint:
        torch.save({"model": model.state_dict()}, f"{save_path}/best.pt")

    print('Best val score: {}'.format(val_curve[best_val_epoch - 1]))
    print('Best test score: {}'.format(test_curve[best_val_epoch - 1]))

    if args.use_wandb:
        wandb.log({f'Best test score, seed {seed}': test_curve[best_val_epoch - 1]})

    logging.info(f"Overall time: {time() - s:.4f}s")
    return test_curve[best_val_epoch - 1]


def main(args):
    checkpoint_folder = "ContactPred" \
                    + "-" + str(args.dataset) \
                    + "-" + str(args.seed) \
                    + "-" + str(args.model) \
                    + "-" + str(args.lr) \
                    + "-" + str(args.batch_size) \
                    + "-" + str(args.clip_norm) \
                    + "-" + str(args.n_layers) \
                    + "-" + str(args.hidden_dim) \
                    + "-" + str(args.drop_ratio) \
                    + "-" + str(args.attn_drop_ratio) \
                    + "-" + str(args.pos_weight) \

    args.save_path = os.path.join(args.save_dir, checkpoint_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)

    if args.dataset == "prot_rna":
        args.complex_type = "prot_rna"

        args.train_esm_path = os.path.join(args.dataset_root, f"split/prot_rna/esm_dict.pkl")
        args.train_fm_path = os.path.join(args.dataset_root, f"split/prot_rna/fm_dict.pkl")
        args.val_esm_path = os.path.join(args.dataset_root, f"split/prot_rna/esm_dict.pkl")
        args.val_fm_path = os.path.join(args.dataset_root, f"split/prot_rna/fm_dict.pkl")
        args.test_esm_path = os.path.join(args.dataset_root, f"split/prot_rna/esm_dict.pkl")
        args.test_fm_path = os.path.join(args.dataset_root, f"split/prot_rna/fm_dict.pkl")

        args.data_path = os.path.join(args.dataset_root, f"split/prot_rna/dataset.pkl")
        args.cache_path = {"train": os.path.join(args.dataset_root, f"split/prot_rna/train_hetero.pkl"),
                           "val": os.path.join(args.dataset_root, f"split/prot_rna/val_hetero.pkl"),
                           "test": os.path.join(args.dataset_root, f"split/prot_rna/test_hetero.pkl")}

    elif args.dataset == "prot_dna":
        args.complex_type = "prot_dna"

        args.train_esm_path = os.path.join(args.dataset_root, f"split/prot_dna/esm_dict.pkl")
        args.train_fm_path = None
        args.val_esm_path = os.path.join(args.dataset_root, f"split/prot_dna/esm_dict.pkl")
        args.val_fm_path = None
        args.test_esm_path = os.path.join(args.dataset_root, f"split/prot_dna/esm_dict.pkl")
        args.test_fm_path = None

        args.data_path = os.path.join(args.dataset_root, f"split/prot_dna/dataset.pkl")
        args.cache_path = {"train": os.path.join(args.dataset_root, f"split/prot_dna/train_hetero.pkl"),
                           "val": os.path.join(args.dataset_root, f"split/prot_dna/val_hetero.pkl"),
                           "test": os.path.join(args.dataset_root, f"split/prot_dna/test_hetero.pkl")}

    elif args.dataset in "prot_prot":
        args.complex_type = "prot_prot"

        args.train_esm_path = os.path.join(args.dataset_root, f"split/dips/esm_dict.pkl")
        args.train_fm_path = None
        args.val_esm_path = os.path.join(args.dataset_root, f"split/dips/esm_dict.pkl")
        args.val_fm_path = None
        args.test_esm_path = os.path.join(args.dataset_root, f"split/dips/esm_dict.pkl")
        args.test_fm_path = None

        args.data_path = os.path.join(args.dataset_root, f"split/dips/dataset.pkl")
        args.cache_path = {"train": os.path.join(args.dataset_root, f"split/dips/train_hetero.pkl"),
                           "val": os.path.join(args.dataset_root, f"split/dips/val_hetero.pkl"),
                           "test": os.path.join(args.dataset_root, f"split/dips/test_hetero.pkl")}

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
        test_metric = run(args.seed, args)
        test_metric_list.append(test_metric)

    mean_f1 = np.mean([m["f1"] for m in test_metric_list])
    std_f1 = np.std([m["f1"] for m in test_metric_list])
    mean_prauc = np.mean([m["pr_auc"] for m in test_metric_list])
    std_prauc = np.std([m["pr_auc"] for m in test_metric_list])

    print(f"Mean F1: {mean_f1}, std F1: {std_f1}")
    print(f"Mean PR-AUC: {mean_prauc}, std PR-AUC: {std_prauc}")

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
    parser.add_argument('--dataset', type=str, default='prot_rna', help="prot_rna, prot_dna, prot_prot")

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
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--edge_hidden_dim', type=int, default=64)
    parser.add_argument('--drop_ratio', type=float, default=0.2)
    parser.add_argument('--attn_drop_ratio', type=float, default=0.2)
    parser.add_argument('--top_k_neighbors', type=int, default=30)
    parser.add_argument('--top_k_neighbors_partner', type=int, default=30)
    parser.add_argument('--embedding_grad_frac', type=float, default=1)
    parser.add_argument('--max_dist', type=float, default=1e5)
    parser.add_argument("--edge_residue", default=True)

    args = parser.parse_args()
    main(args)
