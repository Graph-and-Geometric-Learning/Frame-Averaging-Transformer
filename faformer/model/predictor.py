import torch
import torch.nn as nn
import torch.nn.functional as F

from faformer.data.constant import *
from faformer.model.encoder.transformer import FAFormer
from faformer.model.encoder.config import FAFormerConfig


def get_geo_encoder(args, input_dim, max_seq_len=800):
    if args.model == "faformer":
        return FAFormer(
            FAFormerConfig(
                d_input=input_dim, 
                d_model=args.hidden_dim, 
                d_edge_model=args.edge_hidden_dim,
                activation=args.act,
                n_heads=args.n_heads, 
                n_layers=args.n_layers,
                proj_drop=args.drop_ratio,
                attn_drop=args.attn_drop_ratio,
                n_neighbors=args.top_k_neighbors,
                valid_radius=args.max_dist,
                embedding_grad_frac=args.embedding_grad_frac,
                n_pos=max_seq_len,
            )
        )
    else:
        raise ValueError("Invalid geo framework: {}".format(args.model))


# predict the binding site with mol1 & mol2 sequence + individual structure
class ContactMapPredictor(nn.Module):
    def __init__(self, args):
        super(ContactMapPredictor, self).__init__()

        self.complex_type = args.complex_type

        if self.complex_type == "prot_rna":
            self.mol2_encoder = get_geo_encoder(args, esm_dim, max_seq_len=800)
            self.mol1_encoder = get_geo_encoder(args, fm_dim, max_seq_len=500)

        elif self.complex_type == "prot_dna":
            self.mol2_encoder = get_geo_encoder(args, esm_dim, max_seq_len=800)
            self.mol1_encoder = get_geo_encoder(args, args.hidden_dim, max_seq_len=500)
            self.mol1_res_emb = nn.Embedding(len(NA_ALPHABET), args.hidden_dim, padding_idx=0)

        elif self.complex_type == "prot_prot":
            self.mol2_encoder = get_geo_encoder(args, esm_dim, max_seq_len=800)
            self.mol1_encoder = get_geo_encoder(args, esm_dim, max_seq_len=800)

        else:
            raise ValueError("Invalid complex type: {}".format(self.complex_type))

        self.threshold = args.threshold
        self.binding_site_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def get_distance_map(self, h_mol1, h_mol2):
        # h_mol1: [B, N_mol1, -1], h_mol2: [B, N_mol2, -1]
        B, N_mol1, N_mol2 = h_mol1.shape[0], h_mol1.shape[1], h_mol2.shape[1]
        h_mol1 = h_mol1.unsqueeze(2).expand(-1, -1, N_mol2, -1)  # [B, N_mol1, N_mol2, -1]
        h_mol2 = h_mol2.unsqueeze(1).expand(-1, N_mol1, -1, -1)  # [B, N_mol1, N_mol2, -1]
        return torch.cat([h_mol1, h_mol2], dim=-1)  # [B, N_mol1, N_mol2, -1]  

    def _decenter(self, X, mask):
        # decenter according to the CA position
        # X: [B, N, 14, 3]
        X_ca, mask = X[:, :, 1], mask.unsqueeze(-1)
        center = (X_ca * mask).sum(dim=1) / mask.sum(dim=1)  # [B, 3]
        X_res = X - center.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(-1)  # [B, N, 14, 3]
        return X_res

    def forward(self, mol1, mol2, contact_map):
        # sequence, coordinates, atom types, features
        # S: [B, N], X: [B, N, 14, 3], A: [B, N, 14], h: [B, N, -1]
        S_mol1, X_mol1, A_mol1, h_mol1 = mol1
        S_mol2, X_mol2, A_mol2, h_mol2 = mol2

        B, N_mol1, N_mol2 = S_mol1.shape[0], S_mol1.shape[1], S_mol2.shape[1]
        mol2_mask, mol1_mask = (S_mol2 != 0).float(), (S_mol1 != 0).float()  # [B, N_mol2], [B, N_mol1]

        if self.complex_type == "prot_dna":
            h_mol1 = self.mol1_res_emb(S_mol1)  # [B, N_mol1, -1]

        decenter_X_mol1, decenter_X_mol2 = self._decenter(X_mol1, mol1_mask), self._decenter(X_mol2, mol2_mask)  # [B, N, 3], decenter the structure to avoid label leakage

        h_mol2 = self.mol2_encoder(h_mol2, decenter_X_mol2[:, :, 1], mol2_mask)
        h_mol1 = self.mol1_encoder(h_mol1, decenter_X_mol1[:, :, 1], mol1_mask)
        if isinstance(h_mol1, tuple):
            h_mol2, h_mol1 = h_mol2[0], h_mol1[0]

        pair_h = self.get_distance_map(h_mol1, h_mol2)  # [B, N_mol1, N_mol2, -1]
        pred_binding_site = self.binding_site_predictor(pair_h)  # [B, N_mol1, N_mol2], the score of the binding site

        # contact map generation
        mask = mol1_mask.unsqueeze(2).expand(-1, -1, N_mol2) * mol2_mask.unsqueeze(1).expand(-1, N_mol1, -1)  # [B, N_mol1, N_mol2]
        y_contact = (contact_map < self.threshold).float().cuda()
        y_contact[mask == 0] = 0

        pred_binding_site = pred_binding_site.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        y_contact = y_contact.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        mask = mask.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        return pred_binding_site, y_contact, mask


# predict the binding site on the given protein
class BindSitePredictor(nn.Module):
    def __init__(self, args):
        super(BindSitePredictor, self).__init__()
        
        self.threshold = args.threshold
        self.encoder = get_geo_encoder(args, esm_dim, max_seq_len=800)
        self.binding_site_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def get_distance_map(self, h_mol1, h_mol2):
        # h_mol1: [B, N_mol1, -1], h_mol2: [B, N_mol2, -1]
        B, N_mol1, N_mol2 = h_mol1.shape[0], h_mol1.shape[1], h_mol2.shape[1]
        h_mol1 = h_mol1.unsqueeze(2).expand(-1, -1, N_mol2, -1)  # [B, N_mol1, N_mol2, -1]
        h_mol2 = h_mol2.unsqueeze(1).expand(-1, N_mol1, -1, -1)  # [B, N_mol1, N_mol2, -1]
        return torch.cat([h_mol1, h_mol2], dim=-1)  # [B, N_mol1, N_mol2, -1]  

    def _decenter(self, X):
        # decenter according to the CA position
        # X: [B, N, 14, 3]
        X_ca = X[:, :, 1]
        mask = (X_ca.sum(dim=-1) != 0).unsqueeze(-1)  # [B, N, 1]
        center = (X_ca * mask).sum(dim=1) / mask.sum(dim=1)  # [B, 3]
        X_res = X - center.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(-1)  # [B, N, 14, 3]
        # X_res[mask.unsqueeze(dim=-1).expand(-1, -1, 14, 3) == 0] = 0
        return X_res

    def forward(self, mol, contact_map):
        # sequence, coordinates, atom types, features
        # S: [B, N], X: [B, N, 14, 3], A: [B, N, 14], h: [B, N, -1]
        S_mol, X_mol, A_mol, h_mol = mol

        B, N_mol, mol_mask = S_mol.shape[0], S_mol.shape[1], (S_mol != 0).float()
        decenter_X_mol = self._decenter(X_mol)  # [B, N, 3], decenter the structure to avoid label leakage

        h_mol = self.encoder(h_mol.float(), decenter_X_mol[:, :, 1], mol_mask)
        if isinstance(h_mol, tuple):
            h_mol = h_mol[0]

        pred_binding_site = self.binding_site_predictor(h_mol)  # [B, N_mol1, N_mol2], the score of the binding site

        # contact map generation
        y_contact = contact_map.float().cuda()
        y_contact[mol_mask == 0] = 0

        pred_binding_site = pred_binding_site.reshape(B, -1)  # [B, N_mol]
        y_contact = y_contact.reshape(B, -1)  # [B, N_mol]
        mask = mol_mask.reshape(B, -1)  # [B, N_mol]
        return pred_binding_site, y_contact, mask


class AptamerScreener(nn.Module):
    def __init__(self, args):
        super(AptamerScreener, self).__init__()

        self.complex_type = args.complex_type

        if self.complex_type == "prot_rna":
            self.prot_encoder = get_geo_encoder(args, esm_dim, max_seq_len=1100)
            self.partner_encoder = get_geo_encoder(args, fm_dim, max_seq_len=500)
        else:
            raise ValueError("Invalid complex type: {}".format(self.complex_type))

        self.threshold = args.threshold
        self.binding_site_predictor = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )

    def _decenter(self, X, mask):
        # decenter according to the CA position
        # X: [B, N, 14, 3]
        X_ca, mask = X[:, :, 1], mask.unsqueeze(-1)
        center = (X_ca * mask).sum(dim=1) / mask.sum(dim=1)  # [B, 3]
        X_res = X - center.unsqueeze(1).unsqueeze(1) * mask.unsqueeze(-1)  # [B, N, 14, 3]
        return X_res

    def get_distance_map(self, h_mol1, h_mol2):
        # h_mol1: [B, N_mol1, -1], h_mol2: [B, N_mol2, -1]
        B, N_mol1, N_mol2 = h_mol1.shape[0], h_mol1.shape[1], h_mol2.shape[1]
        h_mol1 = h_mol1.unsqueeze(2).expand(-1, -1, N_mol2, -1)  # [B, N_mol1, N_mol2, -1]
        h_mol2 = h_mol2.unsqueeze(1).expand(-1, N_mol1, -1, -1)  # [B, N_mol1, N_mol2, -1]
        return torch.cat([h_mol1, h_mol2], dim=-1)  # [B, N_mol1, N_mol2, -1]  

    def inference_embs(self, S_mol, X_mol, A_mol, h_mol, kind="prot"):
        B, N_mol = S_mol.shape[0], S_mol.shape[1]
        mol_mask = (S_mol != 0).float()

        decenter_X_mol = self._decenter(X_mol, mol_mask)  # [B, N, 3], decenter the structure to avoid label leakage

        if kind == "prot":
            h_mol = self.prot_encoder(h_mol, decenter_X_mol[:, :, 1], mol_mask)
        else:
            h_mol = self.partner_encoder(h_mol, decenter_X_mol[:, :, 1], mol_mask)
        
        if isinstance(h_mol, tuple):
            h_mol = h_mol[0]
        return h_mol

    def zero_shot_affinity(self, prot_embs, aptamer_mols):
        S_mol, X_mol, A_mol, h_mol = aptamer_mols

        B, N_aptamers, N_prot = S_mol.shape[0], S_mol.shape[1], prot_embs.shape[1]
        aptamer_mask = (S_mol != 0).float()  # [B, N_aptamer]

        aptamer_embs = self.inference_embs(*aptamer_mols, kind="rna")
        prot_embs = prot_embs.expand(B, -1, -1)  # [B, N_mol2, -1]

        pair_h = self.get_distance_map(aptamer_embs, prot_embs)  # [B, N_mol1, N_mol2, -1]
        pred_binding_site = self.binding_site_predictor(pair_h).squeeze(dim=-1)  # [B, N_mol1, N_mol2], the score of the binding site
        pair_mask = aptamer_mask.unsqueeze(-1).expand(-1, -1, N_prot)  # [B, N_mol1, N_mol2]
        pred_binding_site.masked_fill_(~pair_mask.bool(), -1e9)
        return pred_binding_site.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1)  # [B, 1], the max score of the binding site as the crieterion

    def forward(self, mol1, mol2, contact_map):
        S_mol1, X_mol1, A_mol1, h_mol1 = mol1
        S_mol2, X_mol2, A_mol2, h_mol2 = mol2

        B, N_mol1, N_mol2 = S_mol1.shape[0], S_mol1.shape[1], S_mol2.shape[1]
        mol2_mask, mol1_mask = (S_mol2 != 0).float(), (S_mol1 != 0).float()  # [B, N_mol2], [B, N_mol1]

        decenter_X_mol1, decenter_X_mol2 = self._decenter(X_mol1, mol1_mask), self._decenter(X_mol2, mol2_mask)  # [B, N, 3], decenter the structure to avoid label leakage

        h_mol1 = self.partner_encoder(h_mol1, decenter_X_mol1[:, :, 1], mol1_mask)
        h_mol2 = self.prot_encoder(h_mol2, decenter_X_mol2[:, :, 1], mol2_mask)
        if isinstance(h_mol1, tuple):
            h_mol2, h_mol1 = h_mol2[0], h_mol1[0]

        pair_h = self.get_distance_map(h_mol1, h_mol2)  # [B, N_mol1, N_mol2, -1]
        pred_binding_site = self.binding_site_predictor(pair_h)  # [B, N_mol1, N_mol2], the score of the binding site

        # contact map generation
        mask = mol1_mask.unsqueeze(2).expand(-1, -1, N_mol2) * mol2_mask.unsqueeze(1).expand(-1, N_mol1, -1)  # [B, N_mol1, N_mol2]
        y_contact = (contact_map < self.threshold).float().cuda()
        y_contact[mask == 0] = 0

        pred_binding_site = pred_binding_site.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        y_contact = y_contact.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        mask = mask.reshape(B, -1)  # [B, N_mol1 * N_mol2]
        return pred_binding_site, y_contact, mask