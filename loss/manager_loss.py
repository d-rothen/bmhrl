import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.cider import Cider
from model.hrl import BaselineEstimator

def R_t(caption, target):
    first_action = caption[0]
    cdr = cider([first_action], target)
    rewards = [cdr]
    for i in range(len(caption)-1)+2:
        f = cider(caption[:i]) - rewards[-1]
        rewards.append()
        #action detlta cider?


#Remember to freeze the worker
def grad_m(segment, target, b_m, segment_probabilities):
    loss = -(R(segment, target) - b_m ) * torch.sum(torch.log(segment_probabilities))*0#TODO


class ManagerLoss(nn.Module):
    def __init__(self, d_manager_state, gamma) -> None:
        super(ManagerLoss, self).__init__()
        self.manager_baseline = BaselineEstimator(d_manager_state, 1)
        self.gamma = gamma

    def cider_score(self, actions, gts):
        return 0

    def forward(self, segment, worker_states, gts, prev_cider):
        delta_cider = self.cider_score(actions, gts) - prev_cider
        #TODO handle stop gradient on backprop
        baseline = self.worker_baseline(worker_states)

        loss = (delta_cider - baseline) * (action_probs - torch.ones_like(action_probs))
        return loss