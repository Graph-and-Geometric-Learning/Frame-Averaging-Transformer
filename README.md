# Protein-Nucleic Acid Complex Modeling with Frame Averaging Transformer

This is our PyTorch implementation for the paper:

> Tinglin Huang, Zhenqiao Song, Rex Ying, and Wengong Jin (2024). Protein-Nucleic Acid Complex Modeling with Frame Averaging Transformer. [Paper in arXiv](https://arxiv.org/abs/2406.09586). In NeurIPS'2024, Vancouver, Canada, Dec 10-15, 2024.

## Dataset Preparation

Our datasets includes three protein complex datasets and five aptamer datasets. The description of dataset can be found in `dataset/prot_complex/README.md` and `dataset/aptamer/README.md` respectively. The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1GaRe87g2nwbJbkOwqFZr3h9SqIW8bz7V?usp=drive_link).


## Requirements

The code has been tested running under Python 3.10.14. The required packages are as follows:
- pytorch == 2.3.0
- torch_geometric == 2.5.3
- fair-esm == 2.0.0
- rna-fm == 0.2.2
- einops == 0.8.0

Once you finished these installation, please run install the package by running:
```
pip install -e .
```

## Organization

The code is organized as follows:
- `app/`: the main code for training and testing the model
    - `contact_map_train.py`: the pipeline for contact map prediction task
    - `binding_site_train.py`: the pipeline for binding site prediction task
    - `aptamer_aff_train.py`: the pipeline for unsupervised aptamer screening task
- `data/`: the code for data processing
    - `protein_complex_dataset.py`: dataset loading for protein complex
    - `protein_complex_dataloader.py`: dataloader for protein complex
- `model/`
    - `encoder/`: implementation of the Frame Averaging Transformer
    - `predictor.py`: model wrapper for prediction tasks
- `utils/`: utility functions

## Usage

An quick example of using FAFormer to encode a point cloud is as follows:

```
import torch
from faformer.model.encoder.transformer import FAFormer
from faformer.model.encoder.config import FAFormerConfig

model = FAFormer(
    FAFormerConfig(
        d_input=10,  # input feature dimension
        n_layers=2,
        n_neighbors=2, # number of k-nearest neighbors for each point
        n_heads=1,
        d_model=4,  # hidden size
        d_edge_model=4,  # hidden size for edge representation
        norm="layer",
        activation="swiglu",
    )
)
features = torch.randn(2, 5, 10)  # batch size, number of points, feature dimension
coords = torch.rand(2, 5, 3)  # batch size, number of points, 3D coordinates
pad_mask = torch.tensor([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1]]).bool()  # batch size, number of points. True for padding points

features, coords = model(features, coords, pad_mask=pad_mask)
print(features.shape)
print(coords.shape)
```