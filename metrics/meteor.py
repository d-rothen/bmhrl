from nltk.translate.meteor_score import meteor_score
import torch
import numpy as np

def word_from_vector(vocab, indices):
    return np.array([vocab.itos[i] for i in indices])

class MeteorScore():
    def __init__(self, device, vocab, gts, gamma=0.9):
        self.vocab = vocab
        self.hypos = []
        self.scores = []
        self.section_scores = []
        self.gts = np.array(gts)
        self.B = self.gts.shape[0]
        for b in range(self.B):
            self.scores.append([0.0])
            self.section_scores.append([0.0])
            self.hypos.append("")
        self.device = device
        self.gamma = torch.tensor(gamma).to(self.device).float()

    def _discounted_reward(self, meteor_scores):
        scores = torch.tensor(meteor_scores).to(self.device)[:,1:].float()#Cut initial 0 score
        d_B, d_seq = scores.shape

        discounts = self.gamma ** torch.arange(0, d_seq).to(self.device).float()
        discounts = discounts.repeat(d_B, 1).float()
        
        discounted_rewards = discounts * scores
        return torch.sum(discounted_rewards, dim=1)


    def delta_meteor_step(self, indices):
        words = word_from_vector(self.vocab, indices)

        for b in range(self.B):
            self.hypos[b] = self.hypos[b] + " " + words[b]#TODO first whitespace
            last_score = np.array(self.scores[b])[-1]#TODO efficient
            score = meteor_score([self.gts[b]], self.hypos[b]) - last_score
            self.scores[b].append(score)
        
        scores = torch.from_numpy(np.array(self.scores))
        return self._discounted_reward(scores)

    #Called after delta_meteor step finished a section via critic
    #Not immutable! Section score saved on every call
    #secion mask to mark set sections
    #TODO make immutable
    def delta_meteor_section(self, section_mask):
        for b in range(self.B):
            last_score = np.array(self.section_scores[b])[-1]#TODO efficient

            #TODO Check in DEBUG
            if section_mask[b]:
                score = meteor_score([self.gts[b]], self.hypos[b]) - last_score
                self.section_scores[b].append(score)
            else:
                self.section_scores[b].append(last_score)
        
        return self._discounted_reward(self.section_scores)

