import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class HrlAgent(nn.Module):
    def __init__(self, d_voc_size, d_worker_state, d_manager_state, d_joint_features, d_goal, d_word_embedding
    ) -> None:
        super(HrlAgent, self).__init__()
        self.worker = Worker(d_voc_size, d_worker_state=d_worker_state, d_features=d_joint_features, d_goal=d_goal, 
        d_word_embedding=d_word_embedding,)

class GoalNetwork(nn.Module):
    def __init__(self, d_manager_state, d_fc2, d_goal, dropout_p) -> None:
        super(GoalNetwork, self).__init__()

        self.fc1 = nn.Linear(d_manager_state, d_fc2)
        self.fc2 = nn.Linear(d_fc2, d_goal)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, manager_state):
        x = self.fc1(manager_state)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class LowLevelEncoder(nn.Module):
    def __init__(self, d_joint_features, d_low_level_encoder) -> None:
        super(LowLevelEncoder, self).__init__()
        self.bilstm = torch.nn.LSTM(input_size=d_joint_features, hidden_size=d_low_level_encoder, bidirectional=True)

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

class HighLevelEncoder(nn.Module):
    def __init__(self, d_low_level_encoder, d_high_level_encoder) -> None:
        super(HighLevelEncoder, self).__init__()
        self.bilstm = torch.nn.LSTM(input_size=d_low_level_encoder, hidden_size=d_high_level_encoder)

    def forward(self, joint_features):
        output, (h_n, c_n) = self.bilstm(joint_features)
        return output, (h_n, c_n)

class ContextAttention(nn.Module):
    def __init__(self, d_features, d_state, d_model) -> None:
        super(ContextAttention, self).__init__()

        self.W = nn.Linear(d_features, d_model, bias=False)
        self.U = nn.Linear(d_state, d_model, bias=False)
        #TODO check & choose Initializer
        self.bias = nn.Parameter(torch.rand(d_model))

        self.w = nn.parameter(torch.rand(d_model))

    #TODO Batch compatible
    def forward(self, features, hidden_state):
        wh = self.W(features)
        uh = self.U(hidden_state)
        x = wh + uh + self.bias
        x = torch.tanh(x)
        e_t = torch.dot(self.w, x)
        return e_t

class ManagerPolicy(nn.Module):
    def __init__(self, d_features, d_manager_state, d_worker_state, dropout_p) -> None:
        super(ManagerPolicy, self).__init__()
        d_goal = 512
        d_goal_fc2 = 1024
        d_model = 512

        self.noise_mean = 0
        self.noise_std = 0.1

        self.goalie = GoalNetwork(d_manager_state=d_manager_state, d_fc2=d_goal_fc2, d_goal=d_goal, dropout_p=dropout_p)#TODO use cfg params
        self.context_atn = ContextAttention(d_features=d_features, d_state=d_manager_state, d_model=d_model)

        self.manager_lstm = torch.nn.LSTM(input_size=d_model + d_worker_state, hidden_size=d_manager_state)

    def forward(self, features, worker_hidden_state, manager_hidden_state, manager_cell_state, exploration):
        context_attention = self.context_atn(features, manager_hidden_state)
        lstm_in = torch.cat([context_attention, worker_hidden_state], dim=-1)
        output, (h_n, c_n) = self.manager_lstm(lstm_in, (manager_hidden_state, manager_cell_state))
        goal = self.goalie(output)
        
        if exploration:
            noise = torch.normal(size=goal.size(), mean=self.noise_mean, std=self.noise_std)
            goal = goal + noise

        return goal, h_n, c_n

class Worker(nn.Module):
    def __init__(self, voc_size, d_worker_state, d_features, d_context, d_goal, d_word_embedding) -> None:
        super(Worker, self).__init__()
        d_model = 512
        self.context_atn = ContextAttention(d_features=d_features, d_state=d_worker_state, d_model=d_model)
        self.worker_lstm = nn.LSTM(input_size=d_context + d_goal + d_word_embedding, hidden_size=d_worker_state)
        self.classifier = nn.Linear(in_features=d_worker_state, out_features=voc_size)
        self.softmax = nn.Softmax()
        self.baseline_estimator = BaselineEstimator(d_worker_state, 1)

    def forward(self, goal, features, last_action, worker_hidden_state, worker_cell_state):
        context = self.context_atn(features, worker_hidden_state)
        lstm_in = torch.cat([context, goal, last_action])
        output, (h_n, c_n) = self.worker_lstm(lstm_in, (worker_hidden_state, worker_cell_state))
        x = self.classifier(output)

        pi_t = self.softmax(x)
        return pi_t, h_n, c_n

    def select_action(self, pi_t):
        c = Categorical(probs=pi_t)
        actions_taken = c.sample()
        return actions_taken#, c.log_prob(actions_taken)

#Pretraining format [[word_1, .., word_ending_a_segment_a, .., word_n, <eos>], [word_1, .., word_m, <eos>]]
#Labels [[0,...,1_a+1, 0, ..., 1_n+2] (+1 da <start> nicht in word_segment vorhanden), [0, ..., 1_m+2] ]
class Critic(nn.Module):
    def __init__(self, hidden_size):
        #TODO do embedding in forward or pass embedding?
        word_embedding_size = 300
        # input word sequence to determine if end of segment reached
        #Batch first so batch is first dimension for fc
        self.rnn = nn.RNN(input_size=word_embedding_size, hidden_size=hidden_size, batch_first=True)
        #TODO classification für [continue, end] oder nur [end] - paper lässt auf [end] schließen
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, segment):
        output, h_n = self.rnn(segment)
        return self.sigmoid(output)[:,:,0]#TODO bei single class prediction wieder entfernen

#TODO Uncapped linear regressor -> Bias?
class BaselineEstimator(nn.Module):
    def __init__(self, inputSize, outputSize):
        super(BaselineEstimator, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out