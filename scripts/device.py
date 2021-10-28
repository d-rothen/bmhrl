import torch

def get_device(cfg):
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.device_ids[0])
        return torch.device(cfg.device)
    print("! CUDA not available - Using CPU")
    return torch.device("cpu")