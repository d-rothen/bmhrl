import torch

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    Based on OpenAI Spinning Up DDPG
    """

    def __init__(self, batch_size, obs_dim, act_dim, size):
        self.obs_buf = torch.zeros(size, batch_size, obs_dim, dtype=torch.float32)
        self.obs2_buf = torch.zeros(size, batch_size, obs_dim, dtype=torch.float32)
        self.act_buf = torch.zeros(size, batch_size, act_dim, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, batch_size, dtype=torch.float32)
        self.mask_buf = torch.zeros(size, batch_size, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, mask):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.mask_buf[self.ptr] = mask
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = torch.randint(low=0, high=self.size, size=(batch_size,))

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.mask_buf[idxs])
        return {k: v for k,v in batch.items()}