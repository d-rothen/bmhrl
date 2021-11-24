import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.cider import Cider
from model.hrl import BaselineEstimator

class CrossEntropyWarmstart(nn.Module):
    def __init__(self) -> None:
        super(CrossEntropyWarmstart, self).__init__()
    

    def forward(self, actions_probs, gts):
        prob_of_correct_action = torch.gather(actions_probs, dim=-1, index=gts)
        loss = -torch.sum(torch.log(prob_of_correct_action))
        return loss