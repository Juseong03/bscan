from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, auc as auc3, precision_recall_curve
from sklearn.model_selection import train_test_split

import os
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_device(cuda_num: Optional[int] = None) -> str:
    if cuda_num in [0, 1, 2, 3]:
        return f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"
    return "cpu"


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_optimizer(params, opt_name: str, lr: float = 1e-4, w_decay: Optional[float] = None) -> optim.Optimizer:
    weight_decay = 0 if w_decay is None else w_decay

    optimizer_dict = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
    }
    try:
        optimizer_class = optimizer_dict.get(opt_name.lower())
    except AttributeError:
        assert False, f"Optimizer '{opt_name}' not recognized."
        
    if optimizer_class:
        return optimizer_class(params, lr=lr, weight_decay=weight_decay)
    else:
        assert False, f"Optimizer '{opt_name}' not recognized."