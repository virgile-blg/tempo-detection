import torch

def tempo_acc_1(pred_tempo, gt_tempo):
    """Accuracy 1: The percentage of tempo estimates within 4% (the precision window) of the ground-truth tempo.
    """
    return torch.mean((torch.abs(pred_tempo - gt_tempo) < 0.04*gt_tempo).float())

    
# TODO
# Accuracy 2: The percentage of tempo estimates within 4% of either the ground-truth tempo, or half, double, three times, or one third of the ground-truth tempo.
