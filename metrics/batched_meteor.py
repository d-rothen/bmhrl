from nltk.translate.meteor_score import meteor_score
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
import numpy as np

def word_from_vector(vocab, indices):
    return np.array([vocab.itos[i] for i in indices])

class MeteorScorer():
    def _meteor_diff(self, pred, trg, mask):
        last_token = (torch.sum(mask, dim=1)) - 1
        B, L = pred.shape
        rewards = torch.zeros(B, L, dtype=torch.float32)

        for b in torch.arange(B):
            seq_len = last_token[b] + 1
            hypo = word_from_vector(self.vocab, pred[b])

            for l in torch.arange(seq_len):
                partial_hypo = hypo[:l+1]
                rewards[b, l] = meteor_score([trg[b]], self.detokenizer.detokenize(partial_hypo))#TODO try also cutting ref to match hypo? but could overfit

        delta_meteor = rewards[:, 1:] - rewards[:,:-1]
        delta_meteor = torch.cat((rewards[:,0].unsqueeze(-1), delta_meteor), dim=1)

        return delta_meteor

    def segment_reward_queue(self, reward, sections):
        B,L = reward.shape
        segment_reward = torch.zeros(B,L, dtype=torch.float32)
        segment_count = torch.zeros(B, dtype=torch.int)
        
        segment_indices = torch.nonzero(sections)
        
        for segment_idx in segment_indices:
            b,l = segment_idx
            reward_idx = segment_count[b]
            
            segment_reward[b,reward_idx] = reward[b,l]
            segment_count[b] += 1
        return segment_reward, segment_indices

    def delta_meteor_segment(self, delta_meteor_step_reward, sections, gamma_matrix):
        B,L = delta_meteor_step_reward.shape

        segment_reward_queue, segment_reward_index = self.segment_reward_queue(delta_meteor_step_reward, sections).to(self.device)
        discounted_segment_reward = torch.einsum("bl,bsl->bs",segment_reward_queue, gamma_matrix)

        reward = torch.zeros(B,L)
        segment_index = torch.zeros(B, dtype=torch.int32)
        for sr_index in segment_reward_index:
            b,l = sr_index
            reward[b,l] = discounted_segment_reward[b, segment_index[b]]
            segment_index[b] += 1
        return reward

    def delta_meteor_step(self, pred, trg, mask, gamma_matrix):
        meteor_diff = self._meteor_diff(pred, trg, mask).to(self.device)

        discounted_meteor = torch.einsum("bl,bsl->bs",meteor_diff, gamma_matrix)
        return discounted_meteor

    def delta_meteor(self, pred, trg, mask, sections=None):
        B,L = pred.shape
        gamma_matrix = self.get_gamma_matrix(self.gamma, B, L)

        delta_meteor_step_reward = self.delta_meteor_step(pred, trg, mask, gamma_matrix)
        delta_meteor_step_reward = delta_meteor_step_reward

        if sections is not None:
            return self.delta_meteor_segment(delta_meteor_step_reward, sections, gamma_matrix)
        return delta_meteor_step_reward


    def expand_gamma(self, gamma):
        B,L = gamma.shape
        gammas = torch.zeros(B,L,L)
        for j in (np.arange(L)+1)[::-1]:
            index = L-j
            prefix = torch.zeros(L-j, dtype=torch.float32).repeat((B,1))
            res = torch.cat((prefix, gamma[:,:j]), 1)

            gammas[:,index,:] = res
        return gammas

    def get_gamma_matrix(self, gamma, B, L):
        gamma_exp = torch.arange(L, dtype=torch.float32).repeat((B,1))
        gamma_mat = gamma**gamma_exp

        return self.expand_gamma(gamma_mat).to(self.device)

    def __init__(self, vocab, device, gamma_step, gamma_section) -> None:
        self.vocab = vocab
        self.device = device
        self.gamma = gamma_step
        self.detokenizer = TreebankWordDetokenizer()
        #TODO use different gamm for worker/manager