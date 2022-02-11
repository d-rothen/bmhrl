from nltk.translate.meteor_score import meteor_score
import torch
import numpy as np

def word_from_vector(vocab, indices):
    return np.array([vocab.itos[i] for i in indices])

class MeteorCriterion():
    def __init__(self, vocab) -> None:
        self.vocab = vocab

    def __call__(self, gt, pred_indices):
        B,L = pred_indices.shape
        words = word_from_vector(self.vocab, pred_indices)
        score = 0
        for b in range(B):
            score = score + meteor_score([gt[b]], words[b])
        
        return -1/(score/B)#TODO better score metric, this is just used for val next word tho