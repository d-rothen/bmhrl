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
from scripts.device import get_device
import sys
import os
import spacy

class BaselineEstimator(nn.Module):
    def __init__(self, inputSize, outputSize, name):
        super(BaselineEstimator, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.name = name

    def forward(self, x):
        out = self.linear(x)
        return out

class LowLevelEncoder(nn.Module):
    def __init__(self, d_joint_features, d_hidden_state, name="low_level_enc") -> None:
        super(LowLevelEncoder, self).__init__()
        self.name = name
        self.bilstm = torch.nn.LSTM(input_size=d_joint_features, hidden_size=d_hidden_state,
        batch_first=True, bidirectional=True)

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

class HighLevelEncoder(nn.Module):
    def __init__(self, d_low_level_features, d_hidden_state, name="high_level_enc") -> None:
        super(HighLevelEncoder, self).__init__()
        self.name = name
        #2x because bidirectional -> LLEncoder output is 2x d
        self.bilstm = torch.nn.LSTM(input_size=2*d_low_level_features, hidden_size=d_hidden_state,
        batch_first=True, bidirectional=True)#TODO uni directional?

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

#class Worker(nn.Module):
#    def __init__(self, d_features, d_goal) -> None:
#        super(Worker, self).__init__()
#        self.lstm = nn.LSTM(embed_dim, 2*embed_dim, num_layers=4, batch_first=True)



class Worker(nn.Module):
    def __init__(self, voc_size, d_worker_state, d_context, d_goal, d_word_embedding, name="worker") -> None:
        super(Worker, self).__init__()
        self.name = name
        #self.attention = nn.MultiheadAttention

        self.worker_lstm = nn.LSTM(input_size=2*d_context + d_goal + d_word_embedding,
        batch_first=True, hidden_size=d_worker_state)

        self.softmax = nn.Softmax()


        #TODO out shape lower?
        self.projection_1 = nn.Linear(in_features=d_worker_state, out_features=d_worker_state)
        self.tanh = nn.Tanh()
        self.projection_2 = nn.Linear(in_features=d_worker_state, out_features=voc_size)
        self.relu = nn.ReLU()

    def forward(self, goal, low_level_features, last_action):
        #context = self.context_atn(features, worker_hidden_state)
        #TODO attention instead of directly using low level featues
        lstm_in = torch.cat([low_level_features, goal, last_action], -1).unsqueeze(1)#TODO use sequential model
        output, (h_n, c_n) = self.worker_lstm(lstm_in)

        output = torch.squeeze(output)
        x = self.projection_1(output)
        x = self.tanh(x)
        x = self.projection_2(x)

        #x = self.relu(x) TODO this fixxes the neg prob problem? why
        pi_t = self.softmax(x)

        return pi_t, (h_n, c_n)


class Manager(nn.Module):
    def __init__(self, d_features, d_worker_state, d_hidden, d_goal, device, exploration=True, noise_mean=0, noise_std=0.1, name="manager") -> None:
        super(Manager, self).__init__()
        self.name = name
        self.d_goal = d_goal
        self.exploration = exploration
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.device = device

        self.manager_lstm = nn.LSTM(input_size=2*d_features + d_worker_state, hidden_size=d_hidden,
        batch_first=True)

        self.projection_1 = nn.Linear(in_features=d_hidden, out_features=d_hidden)
        self.tanh = nn.Tanh()
        self.projection_2 = nn.Linear(in_features=d_hidden, out_features=d_goal)

    def forward(self, high_level_features, worker_state):#, critic_segments, old_goals):
        lstm_in = torch.cat([high_level_features, worker_state], -1).unsqueeze(1)#TODO use sequential model
        output, (h_n, c_n) = self.manager_lstm(lstm_in)
        #output 0 mu_theta_m
        x = self.projection_1(output)
        x = self.tanh(x)
        x = self.projection_2(x) #g_t

        if self.exploration:
            #TODO triggers Assertion `THCNumerics<T>::ge(val, zero)` failed.
            #TODO or other line https://discuss.pytorch.org/t/help-understand-cuda-error-device-side-assert-triggered/63777
            noise = torch.empty(self.d_goal).normal_(mean=self.noise_mean, std=self.noise_std).to(self.device)
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

        for name, param in self.named_parameters():
            param.requires_grad = False

        
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
    def __init__(self, hidden_state, hidden_agent_state, name) -> None:
        super(ContextAttention, self).__init__()
        self.name = name
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
    def __init__(self, cfg, vocabulary) -> None:
        super(HRLAgent, self).__init__()
        self.device = get_device(cfg)
        #self.inference = False#Set inference mode to false
        self.max_len = cfg.max_len
        self.d_goal = cfg.rl_goal_d
        self.d_w_h = cfg.rl_worker_lstm
        self.d_m_h = cfg.rl_manager_lstm
        self.critic_score_threshhold =cfg.rl_critic_score_threshhold

        self.vocab = vocabulary#TODO remove
        self.voc_size = len(vocabulary)
        embed_dim = cfg.d_model_caps

        self.pad_token = cfg.pad_token
        self.end_token = cfg.end_token
        self.start_token = cfg.start_token
        #TODO important - make sure starting token is 2, this should work for downloaded glove embeddings
        self.pad_index = torch.tensor(1).to(self.device)
        self.start_index = torch.tensor(2).to(self.device)
        self.end_index = torch.tensor(3).to(self.device)

        self.embedding = VocabularyEmbedder(self.voc_size, embed_dim)
        self.embedding.init_word_embeddings(vocabulary.vectors, cfg.unfreeze_word_emb)

        self.worker = Worker(voc_size=self.voc_size, d_worker_state=cfg.rl_worker_lstm, d_goal=self.d_goal, 
        d_word_embedding=cfg.d_model_caps, d_context=cfg.rl_low_level_enc_d,)
        self.worker_baseline_estimator = BaselineEstimator(cfg.rl_worker_lstm, 1, "worker_baseline")
        self.worker_context_atn = ContextAttention(cfg.rl_low_level_enc_d, cfg.rl_worker_lstm, "worker_context_atn")

        self.manager = Manager(d_features=cfg.rl_high_level_enc_d, d_worker_state=cfg.rl_worker_lstm,
        d_hidden=cfg.rl_manager_lstm, d_goal=cfg.rl_goal_d, device=self.device)
        self.manager_baseline_estimator = BaselineEstimator(cfg.rl_manager_lstm, 1, "manager_baseline")
        self.manager_context_atn = ContextAttention(cfg.rl_high_level_enc_d, cfg.rl_manager_lstm, "manager_context_atn")

        self.critic = Critic(embed_dim=cfg.d_model_caps)
        self.critic.load_state_dict(torch.load(cfg.rl_critic_path))

        #TODO Change depending on used modaolities - Here jsut for Video
        d_joint_features = cfg.d_vid
        
        self.low_level_encoder = LowLevelEncoder(d_joint_features=d_joint_features, 
        d_hidden_state=cfg.rl_low_level_enc_d)

        self.high_level_encoder = HighLevelEncoder(d_low_level_features=cfg.rl_low_level_enc_d,
        d_hidden_state=cfg.rl_high_level_enc_d)

    def save_model(self, path):
        #save worker - baseline estimator - context atn, manager baseline estimator - context atn, ll encoder, hl encoder
        self.save_submodule(path, self.worker)
        self.save_submodule(path, self.worker_baseline_estimator)
        self.save_submodule(path, self.worker_context_atn)

        self.save_submodule(path, self.manager)
        self.save_submodule(path, self.manager_baseline_estimator)
        self.save_submodule(path, self.manager_context_atn)

        self.save_submodule(path, self.low_level_encoder)
        self.save_submodule(path, self.high_level_encoder)

    def save_submodule(self, path, module):
        torch.save(module.state_dict(), f'{path}/{module.name}.cp')

    def load_submodule(self, path, module):
        file_path = f'{path}/{module.name}.cp'
        if not os.path.exists(file_path):
            print(f"Did not find checkpoint for {module.name}", file=sys.stderr)
            return False
        module.load_state_dict(torch.load(file_path))
        return True

    def load_model(self, path):
        #save worker - baseline estimator - context atn, manager baseline estimator - context atn, ll encoder, hl encoder
        loaded_model = False

        loaded_model = loaded_model or self.load_submodule(path, self.worker)
        loaded_model = loaded_model or self.load_submodule(path, self.worker_baseline_estimator)
        loaded_model = loaded_model or self.load_submodule(path, self.worker_context_atn)

        loaded_model = loaded_model or self.load_submodule(path, self.manager)
        loaded_model = loaded_model or self.load_submodule(path, self.manager_baseline_estimator)
        loaded_model = loaded_model or self.load_submodule(path, self.manager_context_atn)

        loaded_model = loaded_model or self.load_submodule(path, self.low_level_encoder)
        loaded_model = loaded_model or self.load_submodule(path, self.high_level_encoder)

        return loaded_model

    #def set_inference_mode(self, inference):
    #    self.inference = inference
        
    def cumulative_worker_reward(self, step_rewards, segments):
        B, S = segments.shape
        m, am = torch.max(segments, 1)
        rewards = torch.empty(B)
        for b in range(B):
            from_index = am[b]+1 if m[b] == 1 else 0
            rewards[b]= step_rewards[b,from_index:].sum()
        return rewards.to(self.device)

    def set_freeze_manager_baseline(self, freeze):
        for name, param in self.manager_baseline_estimator.named_parameters():
            param.requires_grad = not freeze

    def set_freeze_worker_baseline(self, freeze):
        for name, param in self.worker_baseline_estimator.named_parameters():
            param.requires_grad = not freeze 

    def set_freeze_worker(self, freeze):
        for name, param in self.worker.named_parameters():
            param.requires_grad = not freeze
        for name, param in self.worker_context_atn.named_parameters():
            param.requires_grad = not freeze

    def set_freeze_manager(self, freeze):
        for name, param in self.manager.named_parameters():
            param.requires_grad = not freeze
        for name, param in self.manager_context_atn.named_parameters():
            param.requires_grad = not freeze

    def warmstart(self, x, gts):
        # x = (B, L, 1024)
        B,L = gts.shape

        low_level_features, (l_h, l_c) = self.low_level_encoder(x)
        high_level_features, (h_h, h_c)  = self.high_level_encoder(low_level_features)

        word_index = 1#Skip <s> token
        goal = torch.zeros(size=(B,self.d_goal)).to(self.device)
        last_w_h = torch.zeros(size=(B,self.d_w_h)).to(self.device)
        last_m_h = torch.zeros(size=(B,self.d_m_h)).to(self.device)

        completion_mask = torch.zeros(B).bool().to(self.device)

        gt_embeddings = self.embedding(gts)
        likelyhood = torch.zeros(size=(B,self.max_len+1)).to(self.device)#+1 for padding of start index

        segments = self.critic(gt_embeddings)
        segments_sm = torch.sigmoid(segments)
        segment_labels = (segments_sm > self.critic_score_threshhold).squeeze().int()

        #TODO finish on full completion mask
        for i in range(self.max_len):
            if i >= (L-1) or torch.all(completion_mask):
                break

            last_word = gt_embeddings[:,word_index-1,:]
            desired_word_index = gts[:,word_index]
            worker_ctx = self.worker_context_atn(low_level_features, last_w_h)

            next_word, (w_h, w_c) = self.worker(goal, worker_ctx, last_word)
            worker_baseline = self.worker_baseline_estimator(w_h).squeeze()#TODO Cut gradient from worker from baseline

            distribution = Categorical(next_word)
            #See how likely groudntruth would have occured
            likelyhood[:,word_index] = torch.gather(distribution.probs, 1, torch.unsqueeze(desired_word_index,-1)).squeeze()#this will start with <s>

            eos = desired_word_index == self.end_index
            completion_mask = completion_mask | eos

            iteration_labels = segment_labels[:,word_index]

            persistent_goal_factor = (~iteration_labels.bool()).int().repeat(self.d_goal, 1).T.float()
            new_goal_factor = iteration_labels.repeat(self.d_goal, 1).T.float()

            persistent_goals = persistent_goal_factor * goal

            manager_ctx = self.manager_context_atn(high_level_features, last_m_h)
            next_goal, (m_h, m_c) = self.manager(manager_ctx, w_h.squeeze())#TODO critic and lastgoal

            last_w_h = w_h.squeeze()
            last_m_h = m_h.squeeze()
            goal = persistent_goals + new_goal_factor * next_goal.squeeze()
            #Append next word(s)
            word_index += 1
        return likelyhood


    def get_worker_weights(self, score, action, eos):
        worker_score = score.delta_meteor_step(action)
        #worker_score_baseline = self.worker_baseline_estimator(w_h).squeeze()#TODO Cut gradient from worker from baseline
        return worker_score * (~eos).float() #Disregard token after </s>

    def get_manager_weights(self, score, worker_weights, segment_labels):
        iteration_labels = segment_labels[:,-1]
        segment_score = score.delta_meteor_section(iteration_labels)
        #manager_baseline = self.manager_baseline_estimator(m_h).squeeze()#TODO Cut gradient from manager from baselin
        #Pass only last segments for accumulation, then only reward segments that terminated this timestep
        cumw = iteration_labels.float() * self.cumulative_worker_reward(worker_weights, segment_labels)
        manager_loss = (segment_score) * cumw
        return manager_loss

    def forward(self, x, gts, gt_strings):
        # x = (B, L, 1024)
        B,L = gts.shape

        score = MeteorScore(self.device, self.vocab, gt_strings)

        low_level_features, (l_h, l_c) = self.low_level_encoder(x)
        high_level_features, (h_h, h_c)  = self.high_level_encoder(low_level_features)

        start_index = torch.Tensor.repeat(self.start_index, B).to(self.device)
        caption = torch.unsqueeze(self.embedding(start_index), 1).to(self.device)
        actions = torch.unsqueeze(start_index, 1).to(self.device)

        word_index = 1#Skip <s> token
        goal = torch.zeros(size=(B,self.d_goal)).to(self.device)
        last_w_h = torch.zeros(size=(B,self.d_w_h)).to(self.device)
        last_m_h = torch.zeros(size=(B,self.d_m_h)).to(self.device)

        completion_mask = torch.zeros(B).bool().to(self.device)

        gt_embeddings = self.embedding(gts)
        likelyhood = torch.zeros(size=(B,self.max_len+1)).to(self.device)#+1 for padding of start index
        worker_weights = torch.zeros(size=(B,self.max_len+1)).to(self.device)
        manager_weights = torch.zeros(size=(B,self.max_len+1)).to(self.device)


        worker_baseline_losses = torch.zeros(size=(B,self.max_len+1)).to(self.device)
        manager_baseline_losses = torch.zeros(size=(B,self.max_len+1)).to(self.device)

        segments = torch.zeros(size=(B,1), dtype=torch.int32).to(self.device)

        #TODO finish on full completion mask
        for i in range(self.max_len):
            if i >= (L-1) or torch.all(completion_mask):
                break

            #Teacher Method - Use last GT as last "predicted" word
            last_word = gt_embeddings[:,word_index-1,:]
            desired_word_index = gts[:,word_index]

            worker_ctx = self.worker_context_atn(low_level_features, last_w_h)

            next_word, (w_h, w_c) = self.worker(goal, worker_ctx, last_word)

            distribution = Categorical(next_word)
            action = distribution.sample()
            eos = action == self.pad_index
            #See how likely groudntruth would have occured
            likelyhood[:,word_index] = torch.gather(distribution.probs, 1, torch.unsqueeze(desired_word_index,-1)).squeeze()#this will start with <s>

            #eos = desired_word_index == self.end_index
            completion_mask = completion_mask | eos

            embedded_words = self.embedding(action)
            caption = torch.cat([caption, torch.unsqueeze(embedded_words, 1)], 1)
            actions = torch.cat([actions, torch.unsqueeze(action, 1)], 1)

            new_segments = self.critic(caption)
            segment_sm = torch.sigmoid(new_segments)
            segment_labels = (segment_sm > self.critic_score_threshhold).squeeze().int()
            #segments = torch.cat([segments, torch.unsqueeze(segment_labels, 1)], 1)

            iteration_labels = segment_labels[:,-1]

            persistent_goal_factor = (~iteration_labels.bool()).int().repeat(self.d_goal, 1).T.float()
            new_goal_factor = iteration_labels.repeat(self.d_goal, 1).T.float()

            persistent_goals = persistent_goal_factor * goal

            manager_ctx = self.manager_context_atn(high_level_features, last_m_h)
            next_goal, (m_h, m_c) = self.manager(manager_ctx, w_h.squeeze())#TODO critic and lastgoal

            # ------ Compute Score ------

            worker_score = score.delta_meteor_step(action)
            #worker_score_baseline = self.worker_baseline_estimator(w_h).squeeze()#TODO Cut gradient from worker from baseline
            worker_weights[:,word_index] = worker_score * (~eos).float() #Disregard token after </s>
            #worker_weights[:,word_index] = worker_score - worker_score_baseline
            #worker_baseline_losses[:, word_index] = (worker_score - worker_score_baseline)**2
            manager_weights[:,word_index] = self.get_manager_weights(score=score, worker_weights=worker_weights[:,:word_index+1],segment_labels=segment_labels[:,:-1])

            # ------ Compute Score End ------

            last_w_h = w_h.squeeze()
            last_m_h = m_h.squeeze()
            goal = persistent_goals + new_goal_factor * next_goal.squeeze()
            #Append next word(s)
            word_index += 1
        return likelyhood, worker_weights, worker_baseline_losses, manager_weights, manager_baseline_losses

    def inference(self, x):
        # x = (B, L, 1024)
        B,_,_ = x.shape

        low_level_features, (l_h, l_c) = self.low_level_encoder(x)
        high_level_features, (h_h, h_c)  = self.high_level_encoder(low_level_features)

        start_index = torch.Tensor.repeat(self.start_index, B).to(self.device)
        caption = torch.unsqueeze(self.embedding(start_index), 1).to(self.device)
        actions = torch.unsqueeze(start_index, 1).to(self.device)

        word_index = 1#Skip <s> token
        goal = torch.zeros(size=(B,self.d_goal)).to(self.device)
        last_w_h = torch.zeros(size=(B,self.d_w_h)).to(self.device)
        last_m_h = torch.zeros(size=(B,self.d_m_h)).to(self.device)

        completion_mask = torch.zeros(B).bool().to(self.device)

        for i in range(self.max_len):
            #HERE might lose score - doesnt know correct length anymore, adjust max length accordingly
            if torch.all(completion_mask):
                break

            last_word = caption[:,word_index-1,:]

            worker_ctx = self.worker_context_atn(low_level_features, last_w_h)

            next_word, (w_h, w_c) = self.worker(goal, worker_ctx, last_word)

            #distribution = Categorical(next_word)
            most_confident_action = torch.argmax(next_word, 1)
            eos = most_confident_action == self.pad_index

            completion_mask = completion_mask | eos

            embedded_words = self.embedding(most_confident_action)
            caption = torch.cat([caption, torch.unsqueeze(embedded_words, 1)], 1)
            actions = torch.cat([actions, torch.unsqueeze(most_confident_action, 1)], 1)

            new_segments = self.critic(caption)
            segment_sm = torch.sigmoid(new_segments)
            segment_labels = (segment_sm > self.critic_score_threshhold).squeeze().int()

            iteration_labels = segment_labels[:,-1]

            persistent_goal_factor = (~iteration_labels.bool()).int().repeat(self.d_goal, 1).T.float()
            new_goal_factor = iteration_labels.repeat(self.d_goal, 1).T.float()

            persistent_goals = persistent_goal_factor * goal

            manager_ctx = self.manager_context_atn(high_level_features, last_m_h)
            next_goal, (m_h, m_c) = self.manager(manager_ctx, w_h.squeeze())#TODO critic and lastgoal

            last_w_h = w_h.squeeze()
            last_m_h = m_h.squeeze()
            goal = persistent_goals + new_goal_factor * next_goal.squeeze()
            word_index += 1

        return {"caption": caption, "actions": actions}