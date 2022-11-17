import torch
import torch.nn.functional as F


# Idea: Predict (Greedy decoder) sentence
# Compute KL div of prediction 
# Compute Biased / Weighted KL Div with prediction (mask out with GT length)
# Show benefit over Biased KL (Lower loss for good prediction)
# Show probs (that influence score -> amplitude?)

abs_error = torch.nn.L1Loss()

def get_top_outliers(hyp, ref, top_k):
    #wordwise_kl = F.kl_div(hyp, ref, reduction='none')# * -1
    wordwise_kl = abs_error(hyp, ref)
    sentencewise_kl = torch.mean(wordwise_kl, dim=-1)
    
    top_kl = torch.topk(sentencewise_kl, k=top_k)
    
    return top_kl[1]#values at 0, indices at 1

def get_threshhold_outliers(hyp, ref, threshhold):
    wordwise_kl = F.kl_div(hyp, ref, reduction='none') * -1
    sentencewise_kl = torch.mean(wordwise_kl, dim=-1)
    
    return (sentencewise_kl > threshhold).nonzero().squeeze()

