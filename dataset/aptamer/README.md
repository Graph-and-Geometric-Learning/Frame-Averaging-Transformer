## Aptamer datasets

This directory contains five protein complex datasets used in the paper:
- `train.pkl`: Protein-RNA complexes for contain map prediction
    - A list of dictionaries, where the description of each key can be found in `faformer/dataset/prot_complex/README.md`. 
- `GFP_eval.pkl`: Aptamer candidates for GFP
- `NELF_eval.pkl`: Aptamer candidates for NELF
- `HNRNPC_eval.pkl`: Aptamer candidates for HNRNPC
- `CHK2_eval.pkl`: Aptamer candidates for CHK2
- `UBLCP1.pkl`: Aptamer candidates for UBLCP1
- `raw/`: Raw data for the aptamer datasets

Each aptamer dataset contains a dictionary of protein target, and two lists of dictionaries for validation and test aptamer candidates. The keys in each aptamer candidate are as follows:
- `pdb`: A string in the format "Target name_aptamer id".
- `partner_seq`: A string of the aptamer sequence.
- `partner_coords`: The predicted structure of aptamer with the shape (L, 14, 3). The second dimension corresponds to 14 specific atoms, as defined in `faformer/data/constant.py`.
- `affinity`: The experimental determined affinity of the aptamer to the target protein.

### Example

```
import pickle

GFP_dataset = pickle.load(open('GFP_eval.pkl', 'rb'))
print(len(GFP_dataset))  # 3, target/validation aptamer/test aptamer
print(GFP_dataset[0].keys())  # dict_keys(['pdb', 'protein_seq', 'protein_coords'])
print(GFP_dataset[1][0].keys())  # dict_keys(['pdb', 'partner_seq', 'partner_coords', 'affinity'])
```