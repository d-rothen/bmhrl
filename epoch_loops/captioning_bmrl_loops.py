import sys
from tqdm import tqdm
import torch

from epoch_loops.captioning_epoch_loops import make_masks
from utilities.captioning_utils import get_lr
import torch.nn as nn


def bmhrl_inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch):
    src = feature_stacks

    caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
    caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
    masks = make_masks(batch['feature_stacks'], caption_idx, modality, pad_idx)

    V, A = src['rgb'] + src['flow'], src['audio']

    prediction = model.module.inference((V,A), caption_idx, masks)

    predicted_words = torch.max(prediction, -1)

    return predicted_words[1]#Tuple -> return indices

def test_print(msg):
    print(msg, file=sys.stderr)

def test_sentence(loader, prediction):
    return " ".join([loader.dataset.train_vocab.itos[i] for i in prediction[0]])

def bmhrl_test(cfg, model, loader):
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase

    epoch = 0
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        src = batch['feature_stacks']

        test_1 = bmhrl_inference(model, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality, None, batch)
        test_print(f'With trg: {test_sentence(loader, test_1)}')

        synthesis = bmhrl_greedy_decoder(model, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality)
        test_print(f'Greedy Decoder: {test_sentence(loader, synthesis)}')
        pass

def bmhrl_greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
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
            V, A = feature_stacks['rgb'] + feature_stacks['flow'], feature_stacks['audio']
            preds = model.module.inference((V,A), trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg


def bmhrl_validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0

    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        with torch.no_grad():
            prediction = model((V,A), caption_idx, None, masks)
            
            #n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()

            loss = criterion(prediction, caption_idx_y) / n_tokens
            val_total_loss += loss
            
    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm

def rl_loss(log_l, reward, mask):
    B,L = mask.shape
    log_l *= reward #TODO High reward -> good output -> should DISCOURAGE high loss?
    _,Ll = log_l.shape
    zero_padding = Ll - L
    padded_loss_mask = torch.nn.functional.pad(mask, (0,zero_padding,0,0), value=0)
    
    log_l = (log_l * padded_loss_mask)[:,1:]#Also get rid of <s> prob
    log_l[log_l != log_l] = 0#TODO trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

    loss = -torch.sum(log_l)

    return loss


def train_bmhrl(cfg, model, loader, optimizer, epoch, criterion, TBoard, train_worker):
    model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    sp = nn.Softplus(10)#TODO outsource

    if train_worker:
        model.module.teach_worker()
        reward_weight = cfg.rl_reward_weight_worker
    else:
        model.module.teach_manager()
        reward_weight = cfg.rl_reward_weight_manager

    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction, reward = model((V,A), caption_idx, batch['captions'], masks)
        
        log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()

        # ----------------old loss-------------------
        #n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        #loss = criterion(prediction, caption_idx_y, reward) / n_tokens
        # ------------------------------------------


        # ------------temp loss--------------
        loss_mask = (batch['caption_data'].caption != 1).float()#TODO use preset token for padding
        reward *= reward_weight
        loss = rl_loss(log_l, reward, loss_mask)
        # -----------------------------------        

        #greedy_decoder(model, batch['feature_stacks'], 30, 2, 3, 1, 'audio_video')
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)

def warmstart_bmhrl(cfg, model, loader, optimizer, epoch, criterion, TBoard):
    model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction = model.module.warmstart((V,A), caption_idx, batch['captions'], masks)

        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(prediction, caption_idx_y) / n_tokens
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

def warmstart_bmhrl_2(cfg, model, loader, optimizer, epoch, criterion, TBoard):
    model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction = model.module.warmstart((V,A), caption_idx, masks)
                
        log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()

        loss_mask = (batch['caption_data'].caption != 1).float()#TODO use preset token for padding
        B,L = loss_mask.shape

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

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)