import numpy as np
# import tensorboardX as tensorboard
import torch
from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader
from epoch_loops.captioning_bmrl_loops import bmhrl_greedy_decoder, bmhrl_inference, bmhrl_test, bmhrl_validation_next_word_loop, train_bmhrl_bl, warmstart_bmhrl_bl
from loss.rl_label_smoothing import RlLabelSmoothing
from metrics.batched_meteor import MeteorScorer
from model.bm_hrl_agent import BMHrlAgent, BMManagerValueFunction, BMWorkerValueFunction
from utilities.learning import adjust_optimizer_lr
from utilities.out_log import print_to_file as print_log

from captioning_datasets.captioning_dataset import ActivityNetCaptionsDataset
from epoch_loops.captioning_epoch_loops import (save_model,
                                                training_loop, training_loop_incremental,
                                                validation_1by1_loop)
from metrics.validation import MeteorCriterion
from epoch_loops.captioning_rl_loops import (rl_training_loop, inference, validation_next_word_loop, warmstart, rl_likelyhood)
from loss.label_smoothing import LabelSmoothing
from model.captioning_module import BiModalTransformer, Transformer
from scripts.device import get_device
from utilities.captioning_utils import average_metrics_in_two_dicts, timer
from utilities.config_constructor import Config
from model.hrl_agent import HRLAgent
from pathlib import Path
from utilities.folders import get_model_checkpoint_dir
import sys


def get_learned_constants(model):
    p = lambda x: print(x, file=sys.stderr)
    for layer in model.module.bm_worker_fus.decoder.layers:
        p(f'Worker av: {layer.a_v_constant}')
    for layer in model.module.bm_manager_fus.decoder.layers:
        p(f'Manager av: {layer.a_v_constant}')


def test_rl_cap(cfg):
    torch.backends.cudnn.benchmark = True    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # preventing PyTorch from allocating memory on the default device (cuda:0) when the desired 
    # cuda id for training is not 0.
    device = get_device(cfg)

    dataset = ActivityNetCaptionsDataset(cfg, 'val_1', get_full_feat=False)


    # make sure that DataLoader has batch_size = 1!
    loader = DataLoader(dataset, collate_fn=dataset.dont_collate)


    #model = HRLAgent(cfg=cfg, vocabulary=train_dataset.train_vocab)
    model = BMHrlAgent(cfg, dataset)
    worker_value_model = BMWorkerValueFunction(cfg)
    manager_value_model = BMManagerValueFunction(cfg)

    #if cfg.optimizer == 'adam':
    #    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
    #                                weight_decay=cfg.weight_decay)

    model.to(device)
    worker_value_model.to(device)
    manager_value_model.to(device)
    if torch.cuda.is_available:
        print("Num dev " + str(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, cfg.device_ids)
        worker_value_model = torch.nn.DataParallel(worker_value_model, cfg.device_ids)
        manager_value_model = torch.nn.DataParallel(manager_value_model, cfg.device_ids)

    
    if cfg.rl_pretrained_model_dir is not None:
        print(f"Looking for pretrained model at {cfg.rl_pretrained_model_dir}", file=sys.stderr)
        loaded_model = model.module.load_model(cfg.rl_pretrained_model_dir)
        loaded_wv_model = worker_value_model.module.load_model(cfg.rl_pretrained_model_dir)
        loaded_mv_model = manager_value_model.module.load_model(cfg.rl_pretrained_model_dir)


    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')
    
    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None


    models = {
        "captioning": (model),
        "worker": (worker_value_model),
        "manager": (manager_value_model)
    }
    get_learned_constants(model)
    bmhrl_test(cfg, models, loader)

def model_info(cfg):
    dataset = ActivityNetCaptionsDataset(cfg, 'val_1', get_full_feat=False)

    #model = HRLAgent(cfg=cfg, vocabulary=train_dataset.train_vocab)
    model = BMHrlAgent(cfg, dataset)
    count_parameters(model)

def count_parameters(model):
    strings = []
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        strings.append(f'{name}\t{params}')
        total_params+=params
    print("\n".join(strings))
    print(f"Total Trainable Params: {total_params}")
    return total_params