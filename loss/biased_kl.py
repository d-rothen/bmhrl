import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def test_print(msg):
    print(msg, file=sys.stderr)

class BiasedKL(nn.Module):
    def __init__(self, label_smoothing, pad_idx):
        super(BiasedKL, self).__init__()

        self.pad_idx = pad_idx
        self.ls = label_smoothing
        self.trg_factor = 1 - self.ls
        self.kl = nn.KLDivLoss(reduction="none")


    def forward(self, pred, trg, biased_trg, biased_offset):
        B, S, V = pred.shape

        trg_ampl = self.trg_factor * (1 - biased_offset).contiguous().view(-1)

        normed_offset = biased_offset * self.trg_factor
        biased_dist = torch.zeros_like(pred)
        biased_dist = torch.scatter(biased_dist, 2, biased_trg.unsqueeze(-1), normed_offset.unsqueeze(-1))

        # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
        prediction = pred.contiguous().view(-1, V)
        target = trg.contiguous().view(-1)

        test_print(f'{target.unsqueeze(-1).shape}, {trg_ampl.unsqueeze(-1).shape}')

        
        # prior (uniform)
        dist = self.ls * torch.ones_like(prediction) / (V - 2)
        # add smoothed ground-truth to prior (args: dim, index, src (value))

        dist.scatter_(1, target.unsqueeze(-1).long(), trg_ampl.unsqueeze(-1)) #Essentially "One Hot" encode traget with .3 (rest is 1/vocsize-1 * .7)
        # make the padding token to have zero probability
        dist[:, self.pad_idx] = 0
        dist = dist + biased_dist.contiguous().view(-1, V)
        # ?? mask: 1 if target == pad_idx; 0 otherwise 
        mask = torch.nonzero(target == self.pad_idx)
        
        if mask.sum() > 0 and len(mask) > 0: #(padded sentences are present)
            # dim, index, val
            dist.index_fill_(0, mask.squeeze(), 0) #set distance 0 where there are padding tokens

        divergence = self.kl(prediction, dist)
        return divergence