from turtle import forward
import torch.nn as nn
import torch.nn.functional as F

class DeltaCider(nn.Module):
    def __init__(self) -> None:
        super(DeltaCider, self).__init__()

    def cider(self, caption, target):
        _, L = caption.shape
        target = target[:, L]
        


    def delta_cider(self, caption):


    def forward(self, caption, prev_cider):
        for 
