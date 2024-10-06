import wandb
import torch
import numpy as np
import random
import argparse


def log_func(save_path, record_dict, use_wandb=False):
    assert isinstance(record_dict, dict)
    if use_wandb:
        wandb.log(record_dict)
    else:
        f=open(save_path, 'a')
        save_str = " ".join(list(map(str, record_dict.values())))
        save_str += " \n"
        f.write(save_str)
        f.close()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True