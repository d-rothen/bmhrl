import torch

def get_device(cfg, set_device=True):
    if torch.cuda.is_available():
        if set_device:
            torch.cuda.set_device(cfg.device_ids[0])
        return torch.device(cfg.device)
    print("! CUDA not available - Using CPU")
    return torch.device("cpu")

def get_device_by_index(index):
    if torch.cuda.is_available():
        torch.cuda.set_device(index)
        return torch.device(index)
    print("! CUDA not available - Using CPU")
    return torch.device("cpu")