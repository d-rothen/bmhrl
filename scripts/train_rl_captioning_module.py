import numpy as np
# import tensorboardX as tensorboard
import torch
from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader
from utilities.out_log import print_to_file as print_log

from datasets.captioning_dataset import ActivityNetCaptionsDataset
from epoch_loops.captioning_epoch_loops import (save_model,
                                                training_loop, training_loop_incremental,
                                                validation_1by1_loop)
from metrics.validation import MeteorCriterion
from epoch_loops.captioning_rl_loops import (rl_training_loop, inference, validation_next_word_loop, warmstart)
from loss.label_smoothing import LabelSmoothing
from model.captioning_module import BiModalTransformer, Transformer
from scripts.device import get_device
from utilities.captioning_utils import average_metrics_in_two_dicts, timer
from utilities.config_constructor import Config
from model.hrl_agent import HRLAgent
from pathlib import Path
import sys

def train_rl_cap(cfg):
    torch.backends.cudnn.benchmark = True    # doing our best to make it replicable
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # preventing PyTorch from allocating memory on the default device (cuda:0) when the desired 
    # cuda id for training is not 0.
    device = get_device(cfg)

    exp_name = cfg.curr_time[2:]

    

    train_dataset = ActivityNetCaptionsDataset(cfg, 'train', get_full_feat=False)
    val_1_dataset = ActivityNetCaptionsDataset(cfg, 'val_1', get_full_feat=False)
    val_2_dataset = ActivityNetCaptionsDataset(cfg, 'val_2', get_full_feat=False)
    meteor_1_criterion = MeteorCriterion(val_1_dataset.train_vocab)
    meteor_2_criterion = MeteorCriterion(val_2_dataset.train_vocab)
    # make sure that DataLoader has batch_size = 1!
    train_loader = DataLoader(train_dataset, collate_fn=train_dataset.dont_collate)

    #TODO uncomment for later - now uses unecessary ram
    val_1_loader = DataLoader(val_1_dataset, collate_fn=val_1_dataset.dont_collate)
    val_2_loader = DataLoader(val_2_dataset, collate_fn=val_2_dataset.dont_collate)


    model = HRLAgent(cfg=cfg, train_dataset=train_dataset)
    
    #TODO Criterion
    #criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    
    #if cfg.optimizer == 'adam':
    #    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
    #                                weight_decay=cfg.weight_decay)
    #elif cfg.optimizer == 'sgd':
    #    optimizer = torch.optim.SGD(model.parameters(), cfg.lr, cfg.momentum,
    #                                weight_decay=cfg.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_reduce_factor, patience=cfg.lr_patience
        )
    else:
        scheduler = None

    model.to(device)
    if torch.cuda.is_available:
        print("Num dev " + str(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, cfg.device_ids)
    
    model.module.load_model(f'{cfg.rl_model_dir}/baseline')

    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Number of Trainable Parameters: {param_num / 1000000} Mil.')
    
    if cfg.to_log:
        TBoard = tensorboard.SummaryWriter(log_dir=cfg.log_path)
        TBoard.add_scalar('debug/param_number', param_num, 0)
    else:
        TBoard = None

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0

    criterion = False

    out_file = "rl.out"

    is_warmstart = True


    for epoch in range(cfg.epoch_num):

        print(f'The best metrict was unchanged for {num_epoch_best_metric_unchanged} epochs.')
        print(f'Expected early stop @ {epoch+cfg.early_stop_after-num_epoch_best_metric_unchanged}')
        print(f'Started @ {cfg.curr_time}; Current timer: {timer(cfg.curr_time)}')
        
        # stop training if metric hasn't been changed for cfg.early_stop_after epochs
        if num_epoch_best_metric_unchanged == cfg.early_stop_after:
            break
        
        # train
        #training_loop_incremental(cfg, model, train_loader, criterion, optimizer, epoch, TBoard)

        ############# Test

        #model.module.set_inference_mode(True)
        # validation (next word)
        #val_1_loss = validation_next_word_loop(
        #    cfg, model, val_1_loader, inference, meteor_1_criterion, epoch, TBoard, exp_name
        #)
        #model.module.set_inference_mode(False)
        ###########


        if is_warmstart:#0:
            print("Warmstarting HRL agent", file=sys.stderr)
            warmstart(cfg, model, train_loader, optimizer, epoch, TBoard)#TODO does it work here?
        else:
            #TODO log here for error?
            rl_training_loop(cfg, model, train_loader, optimizer, epoch, TBoard)
        model.module.set_inference_mode(True)
        # validation (next word)
        val_1_loss = validation_next_word_loop(
            cfg, model, val_1_loader, inference, meteor_1_criterion, epoch, TBoard, exp_name
        )
        val_2_loss = validation_next_word_loop(
            cfg, model, val_2_loader, inference, meteor_2_criterion, epoch, TBoard, exp_name
        )
        val_avg_loss = (val_1_loss + val_2_loss) / 2

        print_log(out_file, f'{val_1_loss} {val_2_loss}')

        if scheduler is not None:
            scheduler.step(val_avg_loss)

        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at or is_warmstart:
            # validation with g.t. proposals
            val_1_metrics = validation_1by1_loop(
                cfg, model, val_1_loader, inference, epoch, TBoard
            )
            val_2_metrics = validation_1by1_loop(
                cfg, model, val_2_loader, inference, epoch, TBoard
            )

            if cfg.to_log:
                # averaging metrics obtained from val_1 and val_2
                metrics_avg = average_metrics_in_two_dicts(val_1_metrics, val_2_metrics)
                metrics_avg = metrics_avg['Average across tIoUs']
                
                TBoard.add_scalar('metrics/meteor', metrics_avg['METEOR'] * 100, epoch)
                TBoard.add_scalar('metrics/bleu4', metrics_avg['Bleu_4'] * 100, epoch)
                TBoard.add_scalar('metrics/bleu3', metrics_avg['Bleu_3'] * 100, epoch)
                TBoard.add_scalar('metrics/precision', metrics_avg['Precision'] * 100, epoch)
                TBoard.add_scalar('metrics/recall', metrics_avg['Recall'] * 100, epoch)
            
                # saving the model if it is better than the best so far
                if best_metric < metrics_avg['METEOR']:
                    best_metric = metrics_avg['METEOR']
                    
                    model_path = f'{cfg.rl_model_dir}/{str(epoch)}'
                    Path(model_path).mkdir(exist_ok=True)
                    model.module.save_model(model_path)# TODO
                    #save_model(cfg, epoch, model, optimizer, val_1_loss, val_2_loss,
                    #           val_1_metrics, val_2_metrics, train_dataset.trg_voc_size)
                    # reset the early stopping criterion
                    num_epoch_best_metric_unchanged = 0
                else:
                    num_epoch_best_metric_unchanged += 1
        model.module.set_inference_mode(False)

        is_warmstart = epoch <= 2 #TODO just for testing metrics



    print(f'{cfg.curr_time}')
    print(f'best_metric: {best_metric}')
    if cfg.to_log:
        TBoard.close()
