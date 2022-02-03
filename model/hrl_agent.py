from dis import dis
from tkinter import PROJECTING
from turtle import forward
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from metrics.meteor import MeteorScore
from torch.utils.data import Dataset
from model.captioning_module import VocabularyEmbedder
from torchtext import data
import spacy

class BaselineEstimator(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(BaselineEstimator, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class LowLevelEncoder(nn.Module):
    def __init__(self, d_joint_features, d_hidden_state) -> None:
        super(LowLevelEncoder, self).__init__()
        self.bilstm = torch.nn.LSTM(input_size=d_joint_features, hidden_size=d_hidden_state,
        batch_first=True, bidirectional=True)

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

class HighLevelEncoder(nn.Module):
    def __init__(self, d_low_level_features, d_hidden_state) -> None:
        super(HighLevelEncoder, self).__init__()
        #2x because bidirectional -> LLEncoder output is 2x d
        self.bilstm = torch.nn.LSTM(input_size=2*d_low_level_features, hidden_size=d_hidden_state,
        batch_first=True, bidirectional=True)

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

#class Worker(nn.Module):
#    def __init__(self, d_features, d_goal) -> None:
#        super(Worker, self).__init__()
#        self.lstm = nn.LSTM(embed_dim, 2*embed_dim, num_layers=4, batch_first=True)



class Worker(nn.Module):
    def __init__(self, voc_size, d_worker_state, d_context, d_goal, d_word_embedding) -> None:
        super(Worker, self).__init__()
        self.attention = nn.MultiheadAttention

        self.worker_lstm = nn.LSTM(input_size=2*d_context + d_goal + d_word_embedding,
        batch_first=True, hidden_size=d_worker_state)

        self.softmax = nn.Softmax()


        #TODO out shape lower?
        self.projection_1 = nn.Linear(in_features=d_worker_state, out_features=d_worker_state)
        self.tanh = nn.Tanh()
        self.projection_2 = nn.Linear(in_features=d_worker_state, out_features=voc_size)



    def forward(self, goal, low_level_features, last_action):
        #context = self.context_atn(features, worker_hidden_state)
        #TODO attention instead of directly using low level featues
        lstm_in = torch.cat([low_level_features, goal, last_action], -1).unsqueeze(1)#TODO use sequential model
        output, (h_n, c_n) = self.worker_lstm(lstm_in)

        output = torch.squeeze(output)
        x = self.projection_1(output)
        x = self.tanh(x)
        x = self.projection_2(x)

        pi_t = self.softmax(x)

        return pi_t, (h_n, c_n)


class Manager(nn.Module):
    def __init__(self, d_features, d_worker_state, d_hidden, d_goal, exploration=True, noise_mean=0, noise_std=0.1) -> None:
        super(Manager, self).__init__()
        self.d_goal = d_goal
        self.exploration = exploration
        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.manager_lstm = nn.LSTM(input_size=2*d_features + d_worker_state, hidden_size=d_hidden,
        batch_first=True)

        self.projection_1 = nn.Linear(in_features=d_hidden, out_features=d_hidden)
        self.tanh = nn.Tanh()
        self.projection_2 = nn.Linear(in_features=d_hidden, out_features=d_goal)

    def forward(self, high_level_features, worker_state):#, critic_segments, old_goals):
        lstm_in = torch.cat([high_level_features, worker_state], -1).unsqueeze(1)#TODO use sequential model
        output, (h_n, c_n) = self.manager_lstm(lstm_in)

        x = self.projection_1(output)
        x = self.tanh(x)
        x = self.projection_2(x)

        if self.exploration:
            noise = torch.empty(self.d_goal).normal_(mean=self.noise_mean, std=self.noise_std)
            x = x + noise

        return x, (h_n, c_n)



class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super(AReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha

class Critic(nn.Module):
    def __init__(self, embed_dim):
        super(Critic, self).__init__()
        self.name = "Critic"

        #TODO Or rnn as in paper?
        self.lstm = nn.LSTM(embed_dim, 2*embed_dim, num_layers=4, batch_first=True)
        self.gru = nn.GRU(2*embed_dim, 2*embed_dim, num_layers=2, batch_first=True)
        self.lin = nn.Linear(2*embed_dim, 1)
        self.relu = AReLU()
        self.relu2 = AReLU()
        
    def forward(self, embedded_indices):
        #Pretrained Critic
        with torch.no_grad():
            h_1, _ = self.lstm(embedded_indices)
            #TODO relu?
            h_1 = self.relu(h_1)
            h_2, _ = self.gru(h_1)

            #TODO out for every h
            #print('h2:', h_2.shape)
            #x = self.relu(h_2)
            h_2 = self.relu2(h_2)
            x = self.lin(h_2)

            return x

class ContextAttention(nn.Module):
    def __init__(self, hidden_state, hidden_agent_state) -> None:
        super(ContextAttention, self).__init__()

        self.tanh = nn.Tanh()
        self.W = nn.Linear(2*hidden_state, 2*hidden_state)
        self.U = nn.Linear(hidden_agent_state,hidden_agent_state)
        self.bias = nn.Parameter(torch.rand(2*hidden_state))

        self.w = nn.Parameter(torch.rand(2*hidden_state))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, encoder_features, agent_features):
        wh = self.W(encoder_features)
        uh = self.U(agent_features).unsqueeze(1)

        e_t = self.w * self.tanh(wh+uh+self.bias)
        #TODO add bias
        a_t = self.softmax(e_t)

        return torch.sum(a_t*encoder_features, 1)


class HRLAgent(nn.Module):
    def __init__(self, cfg, train_dataset) -> None:
        super(HRLAgent, self).__init__()
        self.max_len = cfg.max_len
        self.d_goal = cfg.rl_goal_d
        self.d_w_h = cfg.rl_worker_lstm
        self.d_m_h = cfg.rl_manager_lstm
        self.critic_score_threshhold =cfg.rl_critic_score_threshhold

        vocabulary = train_dataset.train_vocab
        self.vocab = vocabulary#TODO remove
        self.voc_size = len(vocabulary)
        embed_dim = cfg.d_model_caps

        self.pad_token = cfg.pad_token
        self.end_token = cfg.end_token
        self.start_token = cfg.start_token
        #TODO important - make sure starting token is 2, this should work for downloaded glove embeddings
        self.pad_index = torch.tensor(1)
        self.start_index = torch.tensor(2)
        self.end_index = torch.tensor(3)

        self.embedding = VocabularyEmbedder(self.voc_size, embed_dim)
        self.embedding.init_word_embeddings(vocabulary.vectors, cfg.unfreeze_word_emb)

        self.worker = Worker(voc_size=self.voc_size, d_worker_state=cfg.rl_worker_lstm, d_goal=self.d_goal, 
        d_word_embedding=cfg.d_model_caps, d_context=cfg.rl_low_level_enc_d,)
        self.worker_baseline_estimator = BaselineEstimator(cfg.rl_worker_lstm, 1)
        self.worker_context_atn = ContextAttention(cfg.rl_low_level_enc_d, cfg.rl_worker_lstm)

        self.manager = Manager(d_features=cfg.rl_high_level_enc_d, d_worker_state=cfg.rl_worker_lstm,
        d_hidden=cfg.rl_manager_lstm, d_goal=cfg.rl_goal_d)
        self.manager_baseline_estimator = BaselineEstimator(cfg.rl_manager_lstm, 1)
        self.manager_context_atn = ContextAttention(cfg.rl_high_level_enc_d, cfg.rl_manager_lstm)

        self.critic = Critic(embed_dim=cfg.d_model_caps)
        self.critic.load_state_dict(torch.load(cfg.rl_critic_path))

        #TODO Change depending on used modaolities - Here jsut for Video
        d_joint_features = cfg.d_vid
        
        self.low_level_encoder = LowLevelEncoder(d_joint_features=d_joint_features, 
        d_hidden_state=cfg.rl_low_level_enc_d)

        self.high_level_encoder = HighLevelEncoder(d_low_level_features=cfg.rl_low_level_enc_d,
        d_hidden_state=cfg.rl_high_level_enc_d)

    def worker_loss(self, distribution, action):
        #subtract = nn.functional.one_hot(action, self.voc_size) 
        return 

    def cumulative_worker_reward(self, step_rewards, segments):
        B, S = segments.shape
        m, am = torch.max(segments, 1)
        rewards = torch.empty(B)
        for b in range(B):
            from_index = am[b]+1 if m[b] == 1 else 0
            rewards[b]= step_rewards[b,from_index:].sum()
        return rewards

    def forward(self, x, mask, gts):
        # x = (B, L, 1024)
        B,L,_ = x.shape
        score = MeteorScore(self.vocab, gts)

        low_level_features, (l_h, l_c) = self.low_level_encoder(x)
        high_level_features, (h_h, h_c)  = self.high_level_encoder(low_level_features)

        start_index = torch.Tensor.repeat(self.start_index, B)
        caption = torch.unsqueeze(self.embedding(start_index), 1)
        word_index = 0 
        goal = torch.zeros(size=(B,self.d_goal))
        last_w_h = torch.zeros(size=(B,self.d_w_h))
        last_m_h = torch.zeros(size=(B,self.d_m_h))
        
        segments = torch.zeros(size=(B,1), dtype=torch.int32)
        worker_losses = torch.zeros(size=(B,self.max_len))
        manager_losses = torch.zeros(size=(B,self.max_len))

        for i in range(self.max_len):
            last_word = caption[:,word_index,:]
            
            worker_ctx = self.worker_context_atn(low_level_features, last_w_h)

            next_word, (w_h, w_c) = self.worker(goal, worker_ctx, last_word)
            worker_baseline = self.worker_baseline_estimator(w_h).squeeze()#TODO Cut gradient from worker from baseline


            distribution = Categorical(next_word)
            action = distribution.sample()
            word_score = score.delta_meteor_step(action)
            worker_loss = (word_score - worker_baseline) * distribution.log_prob(action)

            worker_losses[:,word_index] = worker_loss

            embedded_words = self.embedding(action)
            caption = torch.cat([caption, torch.unsqueeze(embedded_words, 1)], 1)

            new_segments = self.critic(caption)
            segment_sm = torch.sigmoid(new_segments)
            segment_labels = (segment_sm > self.critic_score_threshhold).squeeze().int()
            #segments = torch.cat([segments, torch.unsqueeze(segment_labels, 1)], 1)

            iteration_labels = segment_labels[:,-1]
            segment_score = score.delta_meteor_section(iteration_labels)

            persistent_goal_factor = (~iteration_labels.bool()).int().repeat(self.d_goal, 1).T.float()
            new_goal_factor = iteration_labels.repeat(self.d_goal, 1).T.float()

            persistent_goals = persistent_goal_factor * goal

            manager_ctx = self.manager_context_atn(high_level_features, last_m_h)
            next_goal, (m_h, m_c) = self.manager(manager_ctx, w_h.squeeze())#TODO critic and lastgoal
            manager_baseline = self.manager_baseline_estimator(m_h).squeeze()#TODO Cut gradient from manager from baseline

            #Only regard reward sums if a segment ended here
            #TODO what if segments change (critic is LSTM!)
            #Pass only last segments for accumulation, then only reward segments that terminated this timestep
            cumw = iteration_labels.float() * self.cumulative_worker_reward(worker_losses[:,:word_index+1], segment_labels[:,:-1])
            manager_loss = -(segment_score - manager_baseline) * cumw
            manager_losses[:,word_index] = manager_loss

            last_w_h = w_h.squeeze()
            last_m_h = m_h.squeeze()
            goal = persistent_goals + new_goal_factor * next_goal.squeeze()
            #Append next word(s)
            word_index += 1
        return caption, worker_losses[:, :word_index], manager_losses[:, :word_index]

