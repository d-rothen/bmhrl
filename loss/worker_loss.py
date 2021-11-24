import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics.cider import Cider
from model.hrl import BaselineEstimator

class WorkerLoss(nn.Module):
    def __init__(self, d_worker_state, gamma) -> None:
        super(WorkerLoss, self).__init__()
        self.worker_baseline = BaselineEstimator(d_worker_state, 1)
        self.gamma = gamma

    def cider_scores(self, actions, gts):
        return 0
    
    def delta_ciders(self, ciders):
        return 0

    #TODO mask out <blank> neg - rewards
    def discounted_reward(self, undiscounted_rewards):
        #rewards should be [batch, seq_len]
        d_B, d_seq = undiscounted_rewards.size()[0], undiscounted_rewards.size[1]
        discounts = self.gamma ** torch.arange(0, d_seq - 1)
        discounts = discounts.repeat(d_B, 1)
        
        discounted_rewards = discounts * undiscounted_rewards
        return torch.sum(discounted_rewards, dim=1)

    def forward(self, actions, action_probs, worker_states, gts):
        ciders = self.cidder_scores(actions, gts)
        delta_ciders = self.delta_ciders(ciders)

        d_B, d_seq = actions.size()[0], actions.size()[1]
        rewards = []#TODO: Save on device, need not save rewards tbh
        losses = []
        for i in range(d_seq):
            deltas = delta_ciders[:,i:]
            d_rewards = self.discounted_reward(deltas)
            rewards.append(d_rewards)
            
            #Compute Baseline
            worker_state = worker_states[:,i]
            #TODO handle stop gradient on backprop
            baseline = self.worker_baseline(worker_state)

            loss = (d_rewards - baseline) * (action_probs[:,i] - torch.ones(d_B))
            
            losses.append(loss)

        return losses, rewards