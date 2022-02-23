from nltk.translate.meteor_score import meteor_score
import torch
import numpy as np

def word_from_vector(vocab, indices):
    return [vocab.itos[i] for i in indices]

class MeteorCriterion():
    def __init__(self, vocab) -> None:
        self.vocab = vocab
    #TODO remove </s> and following
    def __call__(self, gt, pred_indices):
        B,L = pred_indices.shape
        
        score = 0
        for b in range(B):
            words = word_from_vector(self.vocab, pred_indices[b])[1:]#Remove <s> token
            try:
                first_entry_of_eos = words.index('</s>')
                words = words[:first_entry_of_eos]
            except ValueError:
                pass
            score = score + meteor_score([gt[b]], " ".join(words))
        print(score)
        score = score if score != 0 else 0.1 #TODO better handling of zero scores (should never rly happen anyways tho)
        return -1/(score/B)#TODO better score metric, this is just used for val next word tho