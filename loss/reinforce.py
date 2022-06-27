from turtle import forward
from scripts.device import get_device
from metrics.batched_meteor import MeteorScorer, meteor_score
import torch.nn as nn
import torch

class MeteorReinforce(nn.Module):
    def __call__(self, cfg, vocab):
        self.device = get_device(cfg)
        self.vocab = vocab
        self.scorer = MeteorScorer(vocab, self.device, gamma_step=cfg.rl_gamma_worker, gamma_section=cfg.rl_gamma_manager)

    def forward(self, pred, trg, mask):
        with torch.no_grad():
            return self.scorer.delta_meteor(pred, trg, mask)
        

