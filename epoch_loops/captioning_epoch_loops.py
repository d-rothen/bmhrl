import os
import json
from tqdm import tqdm
import torch
import spacy
from time import time

from model.masking import mask
from evaluation.evaluate import ANETcaptions
from captioning_datasets.load_features import load_features_from_npy
from scripts.device import get_device
from utilities.captioning_utils import HiddenPrints, get_lr

def calculate_metrics(
    reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose=True, only_proposals=False
):
    metrics = {}
    PREDICTION_FIELDS = ['results', 'version', 'external_data']
    evaluator = ANETcaptions(
        reference_paths, submission_path, tIoUs, 
        max_prop_per_vid, PREDICTION_FIELDS, verbose, only_proposals)
    evaluator.evaluate()
    
    for i, tiou in enumerate(tIoUs):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    # Print the averages
    
    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))
    
    return metrics

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

    
def save_model(cfg, epoch, model, optimizer, val_1_loss_value, val_2_loss_value, 
               val_1_metrics, val_2_metrics, trg_voc_size):
    
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_1_loss': val_1_loss_value,
        'val_2_loss': val_2_loss_value,
        'val_1_metrics': val_1_metrics,
        'val_2_metrics': val_2_metrics,
        'trg_voc_size': trg_voc_size,
    }
    
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
    
#     path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_cap_model.pt')
    torch.save(dict_to_save, path_to_save)


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

def predicted_action(pred):
    pass

def sentence_from_caption_idx(dataset):
    return [dataset.train_vocab.itos[i] for i in ints]


#Tokens: 1: fill, 2: start, 3: end
def training_loop_incremental(cfg, model, loader, criterion, optimizer, epoch, TBoard):
    model.train()#.cuda()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    device = get_device(cfg, False)

    start_index = 2
    fill_index = 1
    end_index = 3
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        feature_stacks = batch['feature_stacks']
        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices

        B, S = caption_idx.shape

        max_len = S - 1
        current_slice = 1
        #cfg.pad_token

      
        #TODO cfg.batchsize not 32
        trg = (torch.ones(B, 1) * start_index).long().to(device)
        pred_dist = torch.zeros((B, 1, loader.dataset.trg_voc_size)).float().to(device)
        #completeness_mask = torch.zeros(B, 1).byte().to(device)
        
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
    
        while current_slice <= max_len:
        #while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            

            c_idx_slice = caption_idx[:, :current_slice]
            #c_idx_y_slice = caption_idx_y[:, :current_slice]
            #TODO cfg modality - but without subs?
            masks = make_masks(feature_stacks, c_idx_slice, 'audio_video', loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask
            masks['C_mask'] = masks['C_mask'][:,-1]
            pred = model(feature_stacks, c_idx_slice, masks)
            #pred_dist[current_slice-1] = pred[]
            current_slice += 1

            next_dist = torch.clone(pred[:, -1].unsqueeze(1))
            pred_dist = torch.cat([pred_dist, next_dist], dim=1)
            next_word = pred[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            #completeness_mask = completeness_mask | torch.eq(next_word, end_index).byte()
        
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(pred_dist[:, 1:], caption_idx_y) / n_tokens
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)

def training_loop(cfg, model, loader, criterion, optimizer, epoch, TBoard):
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
            


