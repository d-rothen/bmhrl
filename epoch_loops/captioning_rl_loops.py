import os
import json
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

def rl_training_loop(cfg, model, loader, optimizer, epoch, TBoard):
    model.train()#.cuda()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption 
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        video_features = src['rgb'] + src['flow']
        caption, worker_losses, manager_losses = model(video_features[:,:5,:], masks['V_mask'], batch['captions'])
        B,_,_ = video_features.shape
        worker_losses.mean().backward()

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
            
