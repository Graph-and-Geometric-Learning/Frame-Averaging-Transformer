import os
import pickle
import logging

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from faformer.data.constant import *
from faformer.utils.esm_utils import esm_inference_embedding, fm_inference_embedding


def seq2index(table, seq):
    return [table.index(s) for s in seq]


class ContactPredDataset(Dataset):
    def __init__(self, dataset_name, complex_list, hetero_cache_path, protein_esm_path=None, rna_emb_path=None, save_hetero=True, device=0):
        self.data = complex_list
        self.dataset_name = dataset_name

        prot_embedding_dict, rna_embedding_dict = None, None
        if dataset_name == "prot_prot":
            prot_embedding_dict = self.inference_prot_embedding(complex_list, protein_esm_path, include_partner=True, device=device)
        else:
            prot_embedding_dict = self.inference_prot_embedding(complex_list, protein_esm_path, device=device)
            if dataset_name == "prot_rna":
                rna_embedding_dict = self.inference_rna_embedding(complex_list, rna_emb_path, device=device)

        if hetero_cache_path is None or not os.path.exists(hetero_cache_path):
            logging.info("Preprocessing data...")
            self.complex_graphs = self._preprocess(self.data, hetero_cache_path, prot_embedding_dict, rna_embedding_dict, save_hetero=save_hetero)
        else:
            self.complex_graphs = pickle.load(open(hetero_cache_path, 'rb'))

        self.complex_graphs = [g for g in self.complex_graphs if g['protein'].seq.shape[0] < 800]
        if dataset_name == "prot_prot":
            self.complex_graphs = [g for g in self.complex_graphs if g['partner'].seq.shape[0] < 800]
        else:
            self.complex_graphs = [g for g in self.complex_graphs if g['partner'].seq.shape[0] < 500]
        print(f"Total {len(self.complex_graphs)} complexes")

    def inference_prot_embedding(self, data_list, embedding_path, include_partner=False, device=0):
        if os.path.exists(embedding_path):
            embedding_dict = pickle.load(open(embedding_path, 'rb'))
            return embedding_dict
        else:
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

            esm_input = [(f"description", d['protein_seq']) for d in data_list]
            if include_partner:
                esm_input += [(f"description", d['partner_seq']) for d in data_list]
            embedding_dict = esm_inference_embedding(esm_input, device=device)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
            return embedding_dict

    def inference_rna_embedding(self, data_list, embedding_path, device=0):
        if os.path.exists(embedding_path):
            embedding_dict = pickle.load(open(embedding_path, 'rb'))
            return embedding_dict
        else:
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

            fm_input = [(f"description", d['partner_seq']) for d in data_list]
            embedding_dict = fm_inference_embedding(fm_input, device=device)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
            return embedding_dict

    def _preprocess(self, data_list, save_path, esm_embeddings=None, fm_embeddings=None, save_hetero=True):
        """convert data to HeteroData"""

        complex_graphs_all = []
        for idx, complex in enumerate(data_list):

            if complex['protein_coords'].shape[0] <= 1 or complex['partner_coords'].shape[0] <= 1:
                # remove the protein or rna with only one residue or nucleotide
                continue
            if (complex['protein_coords'][:, 1, :].sum(axis=-1) != 0).sum() <= 5 or (complex['partner_coords'][:, 1, :].sum(axis=-1) != 0).sum() <= 5:
                # remove the protein or rna with less than 5 residue or nucleotide coordinates
                continue
            
            complex_graph = HeteroData()

            # protein
            complex_graph['protein'].atom_pos = torch.from_numpy(complex['protein_coords']).float()  # [L, 14, 3]
            complex_graph['protein'].seq = torch.tensor(seq2index(AA_ALPHABET, complex['protein_seq'])).long()
            complex_graph['protein'].x = esm_embeddings[complex['protein_seq']].clone() if esm_embeddings is not None else None
            complex_graph['protein'].atypes = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[AA_ALPHABET.index(s)]] for s in complex['protein_seq']]
            )  # [L, 14]

            complex_graph['partner'].atom_pos = torch.from_numpy(complex['partner_coords']).float()  # [L, 14, 3]
            if self.dataset_name == "prot_prot":
                complex_graph['partner'].seq = torch.tensor(seq2index(AA_ALPHABET, complex['partner_seq'])).long()
                complex_graph['partner'].x = esm_embeddings[complex['partner_seq']].clone()
                complex_graph['partner'].atypes = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[AA_ALPHABET.index(s)]] for s in complex['partner_seq']]
                    )  # [L, 14]
            else:
                complex_graph['partner'].atypes = torch.tensor(
                    [[NA_ATOM_TYPES.index(a) for a in NA_ATOM14[s]] for s in complex['partner_seq']]
                )  # [L, 14]

                if self.dataset_name == "prot_dna":
                    complex_graph['partner'].seq = torch.tensor(seq2index(DNA_NA_ALPHABET, complex['partner_seq'])).long()
                    complex_graph['partner'].x = None
                else:
                    complex_graph['partner'].seq = torch.tensor(seq2index(NA_ALPHABET, complex['partner_seq'])).long()
                    complex_graph['partner'].x = fm_embeddings[complex['partner_seq']].clone()

            complex_graph.contact_map = torch.from_numpy(complex["contact_map"]).float().t()  # [L1, L2] -> [L2, L1]
            if (complex_graph.contact_map < 6).sum() == 0:
                # no contact
                continue

            complex_graphs_all.append(complex_graph)

        if save_hetero:
            with open(save_path, 'wb') as f:
                pickle.dump((complex_graphs_all), f)
        return complex_graphs_all

    def __len__(self):
        return len(self.complex_graphs)

    def __getitem__(self, idx):
        return self.complex_graphs[idx]


class BindSiteDataset(Dataset):
    def __init__(self, dataset_name, complex_list, hetero_cache_path, protein_esm_path=None, save_hetero=True, device=0):
        self.data = complex_list
        self.dataset_name = dataset_name

        prot_embedding_dict = self.inference_prot_embedding(complex_list, protein_esm_path, device=device)

        if hetero_cache_path is None or not os.path.exists(hetero_cache_path):
            logging.info("Preprocessing data...")
            self.complex_graphs = self._preprocess(self.data, hetero_cache_path, prot_embedding_dict, save_hetero=save_hetero)
        else:
            self.complex_graphs = pickle.load(open(hetero_cache_path, 'rb'))

        self.complex_graphs = [g for g in self.complex_graphs if g['protein'].seq.shape[0] < 800]
        self.complex_graphs = [g for g in self.complex_graphs if g['partner'].atom_pos.shape[0] < 500]
        print(f"Total {len(self.complex_graphs)} complexes")

    def inference_prot_embedding(self, data_list, embedding_path, include_partner=False, device=0):
        if os.path.exists(embedding_path):
            embedding_dict = pickle.load(open(embedding_path, 'rb'))
            return embedding_dict
        else:
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)

            esm_input = [(f"description", d['protein_seq']) for d in data_list]
            if include_partner:
                esm_input += [(f"description", d['partner_seq']) for d in data_list]
            embedding_dict = esm_inference_embedding(esm_input, device=device)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
            return embedding_dict

    def _preprocess(self, data_list, save_path, esm_embeddings=None, save_hetero=True):
        """convert data to HeteroData"""
        complex_graphs_all = []
        for idx, complex in enumerate(data_list):
            complex_graph = HeteroData()

            if complex['protein_coords'].shape[0] <= 1 or complex['partner_coords'].shape[0] <= 1:
                # remove the protein or rna with only one residue or nucleotide
                continue
            if (complex['protein_coords'][:, 1, :].sum(axis=-1) != 0).sum() <= 5 or (complex['partner_coords'][:, 1, :].sum(axis=-1) != 0).sum() <= 5:
                # remove the protein or rna with less than 5 residue or nucleotide coordinates
                continue

            # protein1
            complex_graph['protein'].atom_pos = torch.from_numpy(complex['protein_coords']).float()  # [L, 14, 3]
            complex_graph['protein'].seq = torch.tensor(seq2index(AA_ALPHABET, complex['protein_seq'])).long()
            complex_graph['protein'].x = esm_embeddings[complex['protein_seq']].clone() if esm_embeddings is not None else None
            complex_graph['protein'].atypes = torch.tensor(
                [[ATOM_TYPES.index(a) for a in RES_ATOM14[AA_ALPHABET.index(s)]] for s in complex['protein_seq']]
            )  # [L, 14]

            complex_graph['partner'].atom_pos = torch.from_numpy(complex['partner_coords']).float()  # [L, 14, 3]

            # create residue-based contact map
            complex_graph.contact_map = torch.from_numpy(complex["contact_map"]).float().t()
            if (complex_graph.contact_map < 6).sum() == 0:
                # no contact
                continue

            complex_graphs_all.append(complex_graph)

        if save_hetero:
            with open(save_path, 'wb') as f:
                pickle.dump((complex_graphs_all), f)
        return complex_graphs_all

    def __len__(self):
        return len(self.complex_graphs)

    def __getitem__(self, idx):
        return self.complex_graphs[idx]


class AptamerDataset(Dataset):
    def __init__(self, protein, aptamer_list, hetero_cache_path, protein_esm_path=None, rna_emb_path=None, save_hetero=True, device=0):
        prot_embedding_dict, rna_embedding_dict = None, None
        prot_embedding_dict = self.inference_prot_embedding([protein], protein_esm_path, device=device)
        rna_embedding_dict = self.inference_rna_embedding(aptamer_list, rna_emb_path, device=device)

        if hetero_cache_path is None or not os.path.exists(hetero_cache_path):
            logging.info("Preprocessing data...")
            self.protein, self.aptamer = self._preprocess(protein, aptamer_list, hetero_cache_path, prot_embedding_dict, rna_embedding_dict, save_hetero=save_hetero)
        else:
            self.protein, self.aptamer = pickle.load(open(hetero_cache_path, 'rb'))

        print(f"Total {len(self.aptamer)} aptamer")

    def inference_prot_embedding(self, data_list, embedding_path, include_partner=False, device=0):
        if os.path.exists(embedding_path):
            embedding_dict = pickle.load(open(embedding_path, 'rb'))
            return embedding_dict
        else:
            esm_input = [(f"description", d['protein_seq']) for d in data_list]
            if include_partner:
                esm_input += [(f"description", d['partner_seq']) for d in data_list]
            embedding_dict = esm_inference_embedding(esm_input, device=device)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
            return embedding_dict

    def inference_rna_embedding(self, data_list, embedding_path, device=0):
        if os.path.exists(embedding_path):
            embedding_dict = pickle.load(open(embedding_path, 'rb'))
            return embedding_dict
        else:
            fm_input = [(f"description", d['partner_seq']) for d in data_list]
            embedding_dict = fm_inference_embedding(fm_input, device=device)
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding_dict, f)
            return embedding_dict

    def _preprocess(self, protein, aptamer_list, save_path, esm_embeddings=None, fm_embeddings=None, save_hetero=True):
        """convert data to HeteroData"""

        protein_graph = HeteroData()
        protein_graph['graph'].atom_pos = torch.from_numpy(protein['protein_coords']).float()  # [L, 14, 3]
        protein_graph['graph'].seq = torch.tensor(seq2index(AA_ALPHABET, protein['protein_seq'])).long()
        protein_graph['graph'].x = esm_embeddings[protein['protein_seq']].clone() if esm_embeddings is not None else None
        protein_graph['graph'].atypes = torch.tensor(
            [[ATOM_TYPES.index(a) for a in RES_ATOM14[AA_ALPHABET.index(s)]] for s in protein['protein_seq']]
        )  # [L, 14]

        aptamer_graph_list = []
        for idx, complex in enumerate(aptamer_list):
            aptamer_graph = HeteroData()

            aptamer_graph['graph'].atom_pos = torch.from_numpy(complex['partner_coords']).float()  # [L, 14, 3]
            aptamer_graph['graph'].atypes = torch.tensor(
                [[NA_ATOM_TYPES.index(a) for a in NA_ATOM14[s]] for s in complex['partner_seq']]
            )  # [L, 14]
            aptamer_graph['graph'].seq = torch.tensor(seq2index(NA_ALPHABET, complex['partner_seq'])).long()
            aptamer_graph['graph'].x = fm_embeddings[complex['partner_seq']].clone()
            aptamer_graph['graph'].affinity = torch.tensor(complex["affinity"])
            
            aptamer_graph_list.append(aptamer_graph)

        if save_hetero:
            with open(save_path, 'wb') as f:
                pickle.dump([protein_graph, aptamer_graph_list], f)
        return protein_graph, aptamer_graph_list
    
    def get_protein(self):
        return self.protein

    def __len__(self):
        return len(self.aptamer)
    
    def __getitem__(self, idx):
        return self.aptamer[idx]