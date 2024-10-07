## Protein complex datasets

This directory contains three protein complex datasets used in the paper:
- `prot_dna.pkl`: Protein-DNA complexes
- `prot_rna.pkl`: Protein-RNA complexes
- `prot_prot.pkl`: Protein-Protein complexes

Each pickle file contains train, validation and test splits, where each split is represented as a list of dictionaries. Each dictionary includes the following keys:
- `pdb`: A string in the format "PDB ID_protein chain ID_partner chain ID".
- `protein_seq`: The amino acid sequence of the protein.
- `partner_seq`: The sequence of the interacting partner (Protein/RNA/DNA). 
- `protein_coords`: Protein coordinates with the shape (L1, 14, 3). The second dimension corresponds to 14 specific atoms, as defined in `faformer/data/constant.py`. In the validation and test splits, these coordinates are derived from predicted structures using a structure prediction model.
- `partner_coords`: Partner coordinates with the shape (L2, 14, 3). Similar to protein_coords, the second dimension represents the 14 atoms found in `faformer/data/constant.py`. In the validation and test splits, these coordinates are also from predicted structures.
- `kind`: Interaction type (`prot-dna`, `prot-prot`, `prot-rna`).
- `contact_map`: A contact map between the protein and the partner's bound structures, with the shape (L1, L2). Each element represents the distance between residues, calculated as the minimum distance between any two atoms in the respective residues.

There are two additional keys in the validation/test splits:
- `gt_protein_coords`: The ground truth bound coordinates of the protein, with shape (L1, 14, 3).
- `gt_partner_coords`: The ground truth bound coordinates of the partner, with shape (L2, 14, 3).

### Example

```
import pickle

prot_rna_dataset = pickle.load(open('prot_rna.pkl', 'rb'))
print(len(prot_rna_dataset))  # 3, train/validation/test splits
print(prot_rna_dataset[0][0].keys())  # dict_keys(['pdb', 'affinity', 'protein_seq', 'protein_coords', 'partner_seq', 'partner_coords', 'kind', 'contact_map'])
print(prot_rna_dataset[1][0].keys())  # dict_keys(['pdb', 'affinity', 'protein_seq', 'protein_coords', 'partner_seq', 'partner_coords', 'kind', 'contact_map', 'gt_protein_coords', 'gt_partner_coords'])
```
