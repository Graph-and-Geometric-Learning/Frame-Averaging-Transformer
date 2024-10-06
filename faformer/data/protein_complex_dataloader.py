import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from faformer.data.constant import *


class ContactPredDataLoader(DataLoader):
    def __init__(self, complex_dataset, **kwargs):

        self.complex_dataset = complex_dataset
        array = torch.arange(len(complex_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)

    def __collate_fn__(self, batch_idx):
        """make batch of complex graphs"""
        B = len(batch_idx)
        sampled_batch = [self.complex_dataset[i] for i in batch_idx]

        L_max1 = max([len(b['protein'].atom_pos) for b in sampled_batch])
        L_max2 = max([len(b['partner'].atom_pos) for b in sampled_batch])

        # Build the batch
        X_prot, A_prot, S_prot, prot_embs = [], [], [], []
        X_partner, A_partner, S_partner, partner_embs = [], [], [], []
        contact_map = []
        for i, complex in enumerate(sampled_batch):
            n_prot, n_partner = len(complex['protein'].seq), len(complex['partner'].seq)

            X_prot.append(F.pad(complex['protein'].atom_pos, (0, 0, 0, 0, 0, L_max1 - n_prot)))
            S_prot.append(F.pad(complex['protein'].seq, (0, L_max1 - n_prot)))
            A_prot.append(F.pad(complex['protein'].atypes, (0, 0, 0, L_max1 - n_prot)))
            prot_embs.append(F.pad(complex['protein'].x.cuda(), (0, 0, 0, L_max1 - n_prot)))

            X_partner.append(F.pad(complex['partner'].atom_pos, (0, 0, 0, 0, 0, L_max2 - n_partner)))
            S_partner.append(F.pad(complex['partner'].seq, (0, L_max2 - n_partner)))
            A_partner.append(F.pad(complex['partner'].atypes, (0, 0, 0, L_max2 - n_partner)))
            if hasattr(complex['partner'], 'x') and complex['partner'].x is not None:
                partner_embs.append(F.pad(complex['partner'].x.cuda(), (0, 0, 0, L_max2 - n_partner)))
            
            contact_map.append(F.pad(complex.contact_map, (0, L_max1 - n_prot, 0, L_max2 - n_partner)))

        X_prot, A_prot, S_prot, prot_embs = torch.stack(X_prot), torch.stack(A_prot), torch.stack(S_prot), torch.stack(prot_embs)
        X_partner, A_partner, S_partner = torch.stack(X_partner), torch.stack(A_partner), torch.stack(S_partner)
        if not hasattr(complex['partner'], 'x') or complex['partner'].x is None:
            partner_embs = None  # zero is padding
        else:
            partner_embs = torch.stack(partner_embs)
        contact_map = torch.stack(contact_map)

        return (S_partner, X_partner, A_partner, partner_embs), (S_prot, X_prot, A_prot, prot_embs), contact_map


# predict the binding site only on the protein
class BindSitePredDataLoader(DataLoader):
    def __init__(self, complex_dataset, **kwargs):

        self.complex_dataset = complex_dataset
        array = torch.arange(len(complex_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)

    def __collate_fn__(self, batch_idx):
        """make batch of complex graphs"""
        B = len(batch_idx)
        sampled_batch = [self.complex_dataset[i] for i in batch_idx]

        L_max1 = max([len(b['protein'].atom_pos) for b in sampled_batch])

        # Build the batch
        X_prot, A_prot, S_prot, prot_esm_embs = [], [], [], []
        dssp_feats = []

        contact_map = []
        for i, complex in enumerate(sampled_batch):
            n_prot = len(complex['protein'].seq)

            X_prot.append(F.pad(complex['protein'].atom_pos, (0, 0, 0, 0, 0, L_max1 - n_prot)))
            S_prot.append(F.pad(complex['protein'].seq, (0, L_max1 - n_prot)))
            A_prot.append(F.pad(complex['protein'].atypes, (0, 0, 0, L_max1 - n_prot)))
            prot_esm_embs.append(F.pad(complex['protein'].x.cuda(), (0, 0, 0, L_max1 - n_prot)))
            if hasattr(complex, "dssp"):
                dssp_feats.append(F.pad(complex.dssp.cuda(), (0, 0, 0, L_max1 - n_prot)))

            binding_site_labels = ((complex.contact_map < 6).sum(dim=0) != 0).float()  # [n_prot]
            contact_map.append(F.pad(binding_site_labels, (0, L_max1 - n_prot)))

        X_prot, A_prot, S_prot, prot_esm_embs = torch.stack(X_prot), torch.stack(A_prot), torch.stack(S_prot), torch.stack(prot_esm_embs)
        contact_map = torch.stack(contact_map)
        if len(dssp_feats) > 0:
            dssp_feats = torch.stack(dssp_feats)
            prot_esm_embs = torch.cat([prot_esm_embs, dssp_feats], dim=-1)  # [B, L, 1280+14]

        return (S_prot, X_prot, A_prot, prot_esm_embs), contact_map


class AptamerDataLoader(DataLoader):
    def __init__(self, complex_dataset, **kwargs):

        self.complex_dataset = complex_dataset
        array = torch.arange(len(complex_dataset)).long()

        kwargs["collate_fn"] = self.__collate_fn__
        super().__init__(dataset=array, **kwargs)

    def _construct_batch(self, sampled_batch):
        L_max = max([len(b['graph'].atom_pos) for b in sampled_batch])

        # Build the batch
        X, A, S, h = [], [], [], []
        affinity = []
        for i, complex in enumerate(sampled_batch):
            n = len(complex['graph'].seq)

            X.append(F.pad(complex['graph'].atom_pos, (0, 0, 0, 0, 0, L_max - n)))
            S.append(F.pad(complex['graph'].seq, (0, L_max - n)))
            A.append(F.pad(complex['graph'].atypes, (0, 0, 0, L_max - n)))
            
            if hasattr(complex['graph'], 'x') and complex['graph'].x is not None:
                h.append(F.pad(complex['graph'].x.cuda(), (0, 0, 0, L_max - n)))

            if hasattr(complex['graph'], "affinity") and complex['graph'].affinity is not None:
                affinity.append(complex['graph'].affinity)

        X, A, S = torch.stack(X), torch.stack(A), torch.stack(S)
        if not hasattr(complex['graph'], 'x') or complex['graph'].x is None:
            h = None  # zero is padding
        else:
            h = torch.stack(h)

        if len(affinity) > 0:
            affinity = torch.stack(affinity)
            return (S, X, A, h), affinity
        else:
            return S, X, A, h

    def get_protein(self):
        protein = self.complex_dataset.get_protein()
        return self._construct_batch([protein])

    def __collate_fn__(self, batch_idx):
        """make batch of complex graphs"""
        aptamer_batch = self._construct_batch([self.complex_dataset[i] for i in batch_idx])
        return aptamer_batch