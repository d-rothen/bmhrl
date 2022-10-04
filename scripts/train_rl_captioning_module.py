import numpy as np
# import tensorboardX as tensorboard
import torch
from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader
from epoch_loops.captioning_bmrl_loops import bmhrl_greedy_decoder, bmhrl_inference, bmhrl_test, bmhrl_validation_next_word_loop, train_bmhrl, train_bmhrl_bl, warmstart_bmhrl, warmstart_bmhrl_2, warmstart_bmhrl_bl
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


    #model = HRLAgent(cfg=cfg, vocabulary=train_dataset.train_vocab)
    model = BMHrlAgent(cfg, train_dataset)
    worker_value_model = BMWorkerValueFunction(cfg)
    manager_value_model = BMManagerValueFunction(cfg)


    validation_criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)
    warmstart_criterion = LabelSmoothing(cfg.smoothing, train_dataset.pad_idx)

    wv_criterion = torch.nn.MSELoss(reduction='none')
    mv_criterion = torch.nn.MSELoss(reduction='none')

    scorer = MeteorScorer(train_dataset.train_vocab, device, cfg.rl_gamma_worker, cfg.rl_gamma_manager)

    
    #if cfg.optimizer == 'adam':
    #    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
    #                                weight_decay=cfg.weight_decay)

    cap_lr = cfg.rl_cap_warmstart_lr if cfg.rl_warmstart_epochs > 0 else cfg.rl_cap_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=cap_lr, weight_decay=cfg.weight_decay)
    wv_optimizer = torch.optim.Adam(worker_value_model.parameters(), lr=cfg.rl_value_function_lr)
    mv_optimizer = torch.optim.Adam(manager_value_model.parameters(), lr=cfg.rl_value_function_lr)
    
    if cfg.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_reduce_factor, patience=cfg.lr_patience
        )
    else:
        scheduler = None

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

    # keeping track of the best model 
    best_metric = 0
    # "early stopping" thing
    num_epoch_best_metric_unchanged = 0

    is_warmstart = cfg.rl_warmstart_epochs > 0

    alternate_training_switch = False#Start with Manager
    
    learning_rate_validation = False

    models = {
        "captioning": (model, optimizer, warmstart_criterion),
        "worker": (worker_value_model, wv_optimizer, wv_criterion),
        "manager": (manager_value_model, mv_optimizer, mv_criterion)
    }

    #bmhrl_test(cfg, model, train_loader)

    #bmhrl_test(cfg, model, train_loader)

    #metrics_avg = eval_model(cfg, model, (val_2_loader, 0), bmhrl_greedy_decoder, 0, TBoard)
    #print(f"Meteor#{metrics_avg['METEOR']}", file=sys.stderr)
    #return

    log_prefix = "METEOR@?"

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
            print(f"Warmstarting HRL agent #{str(epoch)}", file=sys.stderr)
            #warmstart_bmhrl_2(cfg, model, train_loader, optimizer, epoch, criterion, TBoard)
            warmstart_bmhrl_bl(cfg, models, scorer, train_loader, epoch, log_prefix, TBoard)
        else:
            #TODO log here for error?
            #rl_likelyhood(cfg, model, train_loader, optimizer, epoch, alternate_training_switch, TBoard)#TODO just train worker for now
            train_bmhrl_bl(cfg, models, scorer, train_loader, epoch, log_prefix, TBoard, alternate_training_switch)

        #model.module.set_inference_mode(True)
        
        
        # VALIDATIO?N FOR LEARNING RATE SCHEDULER ------------------------
        
        if learning_rate_validation:
        #val_1_loss = bmhrl_validation_next_word_loop(
        #    cfg, model, val_1_loader, inference, meteor_1_criterion, epoch, TBoard, exp_name
        #)
            val_1_loss = bmhrl_validation_next_word_loop(
                cfg, model, val_1_loader, inference, validation_criterion, epoch, TBoard, exp_name
            )
            val_2_loss = bmhrl_validation_next_word_loop(
                cfg, model, val_2_loader, inference, validation_criterion, epoch, TBoard, exp_name
            )
            val_avg_loss = (val_1_loss + val_2_loss) / 2

            print(f"Validation avg. Loss: {val_avg_loss}", file=sys.stderr)

            if scheduler is not None:
                scheduler.step(val_avg_loss)

        #-------------------------------------------------------------------


        # validation (1-by-1 word)
        if epoch >= cfg.one_by_one_starts_at:# or is_warmstart:

            # validation with g.t. proposals
            metrics_avg = eval_model(cfg, model, (val_1_loader, 0), bmhrl_greedy_decoder, epoch, TBoard)
            log_prefix = f"METEOR@{metrics_avg['METEOR'] * 100}"
            # saving the model if it is better than the best so far
            if best_metric < metrics_avg['METEOR']:
                best_metric = metrics_avg['METEOR']
                
                checkpoint_dir = get_model_checkpoint_dir(cfg, epoch)
                model.module.save_model(checkpoint_dir)
                worker_value_model.module.save_model(checkpoint_dir)
                manager_value_model.module.save_model(checkpoint_dir)

                #save_model(cfg, epoch, model, optimizer, val_1_loss, val_2_loss,
                #           val_1_metrics, val_2_metrics, train_dataset.trg_voc_size)
                # reset the early stopping criterion
                num_epoch_best_metric_unchanged = 0
            else:
                num_epoch_best_metric_unchanged += 1
        #model.module.set_inference_mode(False)

        if is_warmstart and epoch > (cfg.rl_warmstart_epochs - 1):
            is_warmstart = False
            adjust_optimizer_lr(optimizer, cfg.rl_cap_lr)
        alternate_training_switch = not alternate_training_switch


    print(f'{cfg.curr_time}')
    print(f'best_metric: {best_metric}')
    if cfg.to_log:
        TBoard.close()


def eval_model(cfg, model, val_loaders, decoder, epoch, TBoard):
    model.module.set_inference_mode(True)
    val_1_loader, val_2_loader = val_loaders

    val_1_metrics = validation_1by1_loop(
        cfg, model, val_1_loader, decoder, epoch, TBoard
    )
    #val_2_metrics = validation_1by1_loop(
    #    cfg, model, val_2_loader, bmhrl_inference, epoch, TBoard
    #)

    metrics_avg = val_1_metrics
    metrics_avg = metrics_avg['Average across tIoUs']
    
    TBoard.add_scalar('metrics/meteor', metrics_avg['METEOR'] * 100, epoch)
    TBoard.add_scalar('metrics/bleu4', metrics_avg['Bleu_4'] * 100, epoch)
    TBoard.add_scalar('metrics/bleu3', metrics_avg['Bleu_3'] * 100, epoch)
    TBoard.add_scalar('metrics/precision', metrics_avg['Precision'] * 100, epoch)
    TBoard.add_scalar('metrics/recall', metrics_avg['Recall'] * 100, epoch)

    return metrics_avg
            