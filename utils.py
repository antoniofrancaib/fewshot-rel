"""
utils.py

Utility functions used across the project, including metric computations and seeding.
"""

import torch
import numpy as np
import random
from sklearn.metrics import f1_score

def compute_accuracy(logits, labels):
    """
    Computes accuracy given logits and true labels.
    
    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.
        
    Returns:
        accuracy (float): Percentage of correct predictions.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def compute_f1(logits, labels):
    """
    Computes macro F1 score.
    
    Args:
        logits (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.
    
    Returns:
        f1 (float): Macro F1 score.
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return f1_score(labels, preds, average="macro")

def set_seed(seed):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
