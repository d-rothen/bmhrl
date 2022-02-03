import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from metrics.cider import Cider
from model.hrl import BaselineEstimator, Worker

class WorkerLoss(nn.Module):
    def __init__(self) -> None:
        super(WorkerLoss, self)

    def get_policy(logits):
        return Categorical()