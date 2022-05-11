import os
import json
import sys
from tqdm import tqdm
import torch
import spacy
from time import time


from model.masking import mask
from evaluation.evaluate import ANETcaptions
from datasets.load_features import load_features_from_npy
from scripts.device import get_device
from utilities.captioning_utils import HiddenPrints, get_lr

def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if modality == 'video':
        if captions is None:
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
    elif modality == 'audio':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        else:
            masks['A_mask'], masks['C_mask'] = mask(feature_stacks['audio'][:, :, 0], captions, pad_idx)
    elif modality == 'audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
    elif modality == 'subs_audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['S_mask'] = mask(feature_stacks['subs'], None, pad_idx)

    return masks

def greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    #assert model.training is False, 'call model.eval first'

    with torch.no_grad():
        
        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)#TODO Like this we get a mask allowing the WHOLE image + audio sequence ?
            preds = model(feature_stacks, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg

def inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions):
    video_features = feature_stacks['rgb'] + feature_stacks['flow']
    iteration = model(video_features, 0, captions, False)
    return iteration['actions']#TODO or return embedddings?


def warmstart(cfg, model, loader, optimizer, epoch, TBoard):
    model.train()#.cuda()
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        video_features = src['rgb'] + src['flow']
        likelihood = model.module.warmstart(video_features, batch['caption_data'].caption)

        loss_mask = (batch['caption_data'].caption != 1).float()#TODO use preset token for padding
        B,L = loss_mask.shape


        log_l = torch.log(likelihood)
        _,Ll = log_l.shape
        zero_padding = Ll - L
        padded_loss_mask = torch.nn.functional.pad(loss_mask, (0,zero_padding,0,0), value=0)
        
        log_l = (log_l * padded_loss_mask)[:,1:]#Also get rid of <s> prob
        log_l[log_l != log_l] = 0#TODO trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

        loss = -torch.sum(log_l)
        #print(f"CEL: {loss.numpy()}", file=sys.stderr)
        #loss = loss * 1e-2#* 1e-3#TODO control lr from calling function - here just squeeze the loss a bit to account for high lr
        loss.backward()

        optimizer.step()

        #if i > 100:
            #break
        #manager_losses.mean().backward()

def rl_likelyhood(cfg, model, loader, optimizer, epoch, train_worker, TBoard):
    model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    if train_worker:
        model.module.set_freeze_manager(True)
        model.module.set_freeze_worker(False)
    else:
        model.module.set_freeze_manager(False)
        model.module.set_freeze_worker(True)

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        video_features = src['rgb'] + src['flow']
        likelihood, worker_weight, worker_baseline_loss, manager_weight, manager_baseline_loss = model.module.forward_likelyhood(video_features, batch['caption_data'].caption, batch['captions'], train_worker)

        loss_mask = (batch['caption_data'].caption != 1).float()#TODO use preset token for padding
        B,L = loss_mask.shape


        log_l = torch.log(likelihood)
        _,Ll = log_l.shape
        zero_padding = Ll - L
        padded_loss_mask = torch.nn.functional.pad(loss_mask, (0,zero_padding,0,0), value=0)
        
        if train_worker:
            log_l = log_l * worker_weight

            log_l = (log_l * padded_loss_mask)[:,1:]#Also get rid of <s> prob
            log_l[log_l != log_l] = 0#TODO trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

            loss = -torch.sum(log_l)
            #loss = loss * 1e-2#* 1e-3#TODO control lr from calling function - here just squeeze the loss a bit to account for high lr


            worker_baseline_loss.mean().backward(retain_graph=True)
            model.module.set_freeze_worker_baseline(True)
            loss.backward()
            model.module.set_freeze_worker_baseline(False)
        else:
            pass


        optimizer.step()

        #if i > 100:
            #break
        #manager_losses.mean().backward()

def rl_training_loop(cfg, model, loader, optimizer, epoch, train_worker, TBoard):
    model.train()#.cuda()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    if train_worker:
        model.module.set_freeze_manager(True)
        model.module.set_freeze_worker(False)
    else:
        model.module.set_freeze_manager(False)
        model.module.set_freeze_worker(True)
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption 
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        video_features = src['rgb'] + src['flow']
        iteration = model(video_features, masks['V_mask'], batch['captions'], train_worker)

        if train_worker:##TODO Order relevant? Also does Critic work correctly with vector_cache?
            iteration['worker_baseline_loss'].mean().backward(retain_graph=True)#TODO check cutoffs
            
            model.module.set_freeze_worker_baseline(True)
            worker_loss = iteration["worker_loss"].mean()
            worker_loss.backward()
            model.module.set_freeze_worker_baseline(False)
            
            #print(worker_loss)
        else:
            iteration['manager_baseline_loss'].mean().backward(retain_graph=True)#TODO check cutoffs
            
            model.module.set_freeze_manager_baseline(True)
            manager_loss = iteration["manager_loss"].mean()
            manager_loss.backward()
            model.module.set_freeze_manager_baseline(False)

        optimizer.step()
        #manager_losses.mean().backward()

def training_loop_rl(cfg, model, loader, criterion, optimizer, epoch, TBoard):
    model.train()#.cuda()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask
        pred = model(batch['feature_stacks'], caption_idx, masks)
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(pred, caption_idx_y) / n_tokens
        loss.backward()

        #greedy_decoder(model, batch['feature_stacks'], 30, 2, 3, 1, 'audio_video')

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)
            
def validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        src = batch['feature_stacks']
        video_features = src['rgb'] + src['flow']
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model(video_features, masks['V_mask'], batch['captions'],  False)
            predicted_caption = pred["actions"]
            #n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(batch['captions'], predicted_caption)
            val_total_loss += loss
            
    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm