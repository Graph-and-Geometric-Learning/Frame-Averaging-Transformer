import esm
import fm
import pickle
import torch
import numpy as np
from tqdm import tqdm


@torch.no_grad()
def esm_inference_embedding(esm_input, esm_root_path=None, device=0):
    if esm_root_path is not None:
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(esm_root_path)
    else:
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    n_layers = len(model.layers)

    model.eval().to(device)
    token_reprs = {}  # {protein_seq: embedding}
    for s in tqdm(range(len(esm_input))):
        batch_labels, batch_strs, batch_tokens = batch_converter(esm_input[s:s+1])
        results = model(batch_tokens.to(device), repr_layers=[n_layers], return_contacts=False)
        token_reprs[esm_input[s][1]] = results["representations"][n_layers][:, 1:-1, :].squeeze(0).cpu()  # [seq_len, 1280]
    return token_reprs


@torch.no_grad()
def fm_inference_embedding(esm_input, device=0):
    model, alphabet = fm.pretrained.rna_fm_t12()

    batch_converter = alphabet.get_batch_converter()
    n_layers = len(model.layers)

    model.eval().to(device)
    token_reprs = {}
    for s in tqdm(range(len(esm_input))):
        batch_labels, batch_strs, batch_tokens = batch_converter(esm_input[s:s+1])
        results = model(batch_tokens.to(device), repr_layers=[n_layers], return_contacts=False)
        token_reprs[esm_input[s][1]] = results["representations"][n_layers][:, 1:-1, :].squeeze(0).cpu()  # [seq_len, 640]
    return token_reprs