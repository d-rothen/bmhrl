import sys
from tqdm import tqdm
import traceback
import torch
from model.masking import c_mask
from torch.distributions.categorical import Categorical


from utilities.captioning_utils import get_lr
import torch.nn as nn
import torch.nn.functional as F
from loss.biased_kl import BiasedKL
from loss.label_smoothing import LabelSmoothing
from scripts.device import get_device
from utilities.analyze import get_top_outliers


from model.masking import make_masks
from torch import autograd


def bmhrl_inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch):
    inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch, feature_getter(True, False))

def audio_inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch):
    inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch, feature_getter(False, True))

def audio_inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch):
    inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch, feature_getter(False, False))

def bimodal_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    return greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, inference_feature_getter(True, False))

def video_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    return greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, inference_feature_getter(False, False))

def audio_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    return greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, inference_feature_getter(False, True))


def greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, inference_feature_getter):
    with torch.no_grad():
        B, _Sa_, _Da_ = feature_stacks['audio'].shape
        device = feature_stacks['audio'].device

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            modalities, masks = inference_feature_getter(trg, feature_stacks, modality, pad_idx)

            preds = model.inference(modalities, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg
    
def bmhrl_greedy_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
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
            preds = model.inference((V,A), trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg

def test_print(msg):
    print(msg, file=sys.stderr)

def test_sentence(loader, prediction):
    return " ".join([loader.dataset.train_vocab.itos[i] for i in prediction])

def bmhrl_test(cfg, models, loader):
    cap_model = models["captioning"]
    wv_model = models["worker"]
    mv_model = models["manager"]


    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase

    epoch = 0
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        src = batch['feature_stacks']
        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx_y = caption_idx[:, 1:]
        test_print(f'Groundtruth: {test_sentence(loader, caption_idx_y[0])}')

        test_1 = bimodal_inference(cap_model, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality, None, batch)
        test_print(f'With trg: {test_sentence(loader, test_1[0])}')

        synthesis = bmhrl_greedy_decoder(cap_model.module, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality)
        test_print(f'Greedy Decoder: {test_sentence(loader, synthesis[0])}')



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
            prediction = model((V,A), caption_idx, masks)[0]
            
            #n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()

            loss = criterion(prediction, caption_idx_y) / n_tokens
            val_total_loss += loss
            
    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm

def get_score(train_worker, scorer, predicted_tokens, caption, mask, segments):
    if train_worker:
        return scorer.delta_meteor_worker(predicted_tokens, caption, mask)
    return scorer.delta_meteor_manager(predicted_tokens, caption, mask, segments)


def sample_loss_kl(train_worker, prediction, sampled_prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, biased_kldiv):
    pred_probs = torch.exp(prediction)
    dist = Categorical(pred_probs)
    sampled_prediction = dist.sample()

    score, r = get_score(train_worker, scorer, sampled_prediction, trg_caption, mask, segments)

    if not train_worker:
        log_probs = torch.gather(prediction, 2, sampled_prediction.unsqueeze(-1)).squeeze()
        loss = -log_probs * score
        return loss, [score], [sampled_prediction], [0]

    sampled_probs = torch.gather(pred_probs, 2, sampled_prediction.unsqueeze(-1)).squeeze()

    test_print(f"\nProbs. : min = {torch.min(sampled_probs)}, max = {torch.max(sampled_probs)}")

    norm_reward_factor = torch.sum(mask, dim=-1)
    norm_reward_factor = torch.reshape(norm_reward_factor, (-1,1))

    test_print(f"NormFac. : min = {torch.min(norm_reward_factor)}, max = {torch.max(norm_reward_factor)}") 

    score = torch.clamp(score, 0, 1)
    amplitude = score * sampled_probs * norm_reward_factor.float() 
    #amplitude = torch.clamp(amplitude, 0, 1)#TODO make use of negative rewards

    test_print(f"Amplitude : min = {torch.min(amplitude)}, mean = {torch.mean(amplitude)}, max = {torch.max(amplitude)}") 

    test_print(f'{prediction.shape}, {trg.shape}, {sampled_prediction.shape}, {amplitude.shape}')

    divergence = biased_kldiv(prediction, trg, sampled_prediction, amplitude)

    test_print(f"Divergence. : min = {torch.min(divergence)}, max = {torch.max(divergence)}") 

    return divergence, [score], [sampled_prediction], [amplitude]


def biased_kl(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, biased_kldiv, stabilize):
    pred_probs = torch.exp(prediction)#TODO check
    dist = Categorical(pred_probs)
    sampled_prediction = dist.sample()

    sampled_probs = torch.gather(pred_probs, 2, sampled_prediction.unsqueeze(-1)).squeeze()

    score, rewards = get_score(train_worker, scorer, sampled_prediction, trg_caption, mask, segments)
    score = score.to(device)

    if stabilize:
        score = score - (expected_scores * mask.float())
    
    if not train_worker:
        score = score * segments.float()

    test_print(f"\nProbs. : min = {torch.min(sampled_probs)}, max = {torch.max(sampled_probs)}")

    norm_reward_factor = get_norm_reward_factor(train_worker, mask, segments)

    test_print(f"NormFac. : min = {torch.min(norm_reward_factor)}, max = {torch.max(norm_reward_factor)}") 

    amplitude = get_amplitude(score, sampled_probs, norm_reward_factor)

    test_print(f"Amplitude : min = {torch.min(amplitude)}, mean = {torch.mean(amplitude)}, max = {torch.max(amplitude)}") 

    test_print(f'{prediction.shape}, {trg.shape}, {sampled_prediction.shape}, {amplitude.shape}')

    divergence = biased_kldiv(prediction, trg, sampled_prediction, amplitude)

    test_print(f"Divergence. : min = {torch.min(divergence)}, max = {torch.max(divergence)}") 

    return divergence, [score], [sampled_prediction], [amplitude]

def sum_loss_over_words(loss, B,L):
    loss = torch.reshape(loss, (B,L,-1))
    return torch.sum(loss, dim=2)

def w_b_n_kl(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, biased_kldiv, kldiv, norm_factor, greedy=True):
    pred_probs = torch.exp(prediction)#TODO check

    if greedy:
        sampled_probs, sampled_prediction = pred_probs.max(dim=-1)
    else:
        dist = Categorical(pred_probs)
        sampled_prediction = dist.sample()
        sampled_probs = torch.gather(pred_probs, 2, sampled_prediction.unsqueeze(-1)).squeeze()

    score, rewards = get_score(train_worker, scorer, sampled_prediction, trg_caption, mask, segments)
    norm_reward_factor = get_norm_reward_factor(train_worker, mask, segments)

    amplitude = get_amplitude(score, sampled_probs, norm_reward_factor)
    weighted_amplitude = get_weighted_amplitude(amplitude, norm_factor)

    biased_divergence = biased_kldiv(prediction, trg, sampled_prediction, amplitude)
    divergence = kldiv(prediction, trg)
    weighted_divergence = divergence / weighted_amplitude

    B,L,_ = prediction.shape
    weighted_divergence = sum_loss_over_words(weighted_divergence, B,L)
    biased_divergence = sum_loss_over_words(biased_divergence,B,L)
    divergence = sum_loss_over_words(divergence, B,L)

    return [weighted_divergence, biased_divergence, divergence], [amplitude, score, rewards], [sampled_prediction, sampled_probs]

def get_amplitude(score, sampled_probs, norm_reward_factor):
    amplitude = score * sampled_probs * norm_reward_factor.float()
    return torch.clamp(amplitude, 0, 1)

def get_norm_reward_factor(train_worker, mask, segments):
    norm_reward_factor = torch.sum(mask, dim=-1) if train_worker else torch.sum(segments, dim=-1)
    return torch.reshape(norm_reward_factor, (-1,1))

def get_weighted_amplitude(base_amplitude, norm_factor):
    threshhold = 1/norm_factor
    amplitude = torch.clamp(base_amplitude, min=threshhold, max=1)
    amplitude[amplitude < threshhold] = threshhold

    return amplitude.reshape(-1).unsqueeze(-1)

def weighted_kl(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, kl_div, norm_factor):
    pred_probs = torch.exp(prediction)#TODO check
    dist = Categorical(pred_probs)
    cat_sampled_prediction = dist.sample()

    cat_sampled_probs = torch.gather(pred_probs, 2, cat_sampled_prediction.unsqueeze(-1)).squeeze()
    #log_probs = torch.gather(prediction, 2, cat_sampled_prediction.unsqueeze(-1)).squeeze()
    sampled_probs = cat_sampled_probs
    sampled_prediction_y = cat_sampled_prediction

    score, r = get_score(train_worker, scorer, sampled_prediction_y, trg_caption, mask, segments)

    test_print(f"\nProbs. : min = {torch.min(sampled_probs)}, max = {torch.max(sampled_probs)}")

    norm_reward_factor = get_norm_reward_factor(train_worker, mask, segments)

    amplitude = weighted_amplitude(score, sampled_probs, norm_factor, norm_reward_factor)

    test_print(f"Amplitude : min = {torch.min(amplitude)}, mean = {torch.mean(amplitude)}, max = {torch.max(amplitude)}") 

    test_print(f'{prediction.shape}, {trg.shape}, {sampled_prediction_y.shape}, {amplitude.shape}')

    divergence = kl_div(prediction, trg) / amplitude

    test_print(f"Divergence. : min = {torch.min(divergence)}, max = {torch.max(divergence)}") 

    return divergence, [score], [sampled_prediction_y], [amplitude]

def log_iteration(loader, pred, trg, score, score_pred, amplitude, segments, train_worker):
    B,L = pred.shape
    test_print(f'Summed Score: {torch.sum(score)}')

    for b in range(B):
        test_print(f'Pred[{b}]: {test_sentence(loader, pred[b])}')
        test_print(f'Trg[{b}]: {test_sentence(loader, trg[b])}')
        test_print(f'Score[{b}]: {score[b]}')
        test_print(f'Score_pred[{b}]: {score_pred[b]}')
        test_print(f'Segm[{b}]: {segments[b]}')

def inference_feature_getter(both, audio):
    def get_features(trg, feature_stacks, modality, pad_idx):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)
            V, A = feature_stacks['rgb'] + feature_stacks['flow'], feature_stacks['audio']

            if both:
                return (V,A), masks
            elif audio:
                return A, (masks["A_mask"], masks["C_mask"])
            else:
                return V, (masks["V_mask"], masks["C_mask"])
    return get_features

def feature_getter(both, audio):
    def get_features(cfg, batch, loader):
        src = batch['feature_stacks']
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]

        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)
        V, A = src['rgb'] + src['flow'], src['audio']

        if both:
            return caption_idx, caption_idx_y, (V,A), masks
        elif audio:
            return caption_idx, caption_idx_y, A, (masks["A_mask"], masks["C_mask"])
        else:
            return caption_idx, caption_idx_y, V, (masks["V_mask"], masks["C_mask"])
    return get_features



def train_video_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker):
    return train_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter(False, False))

def train_audio_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker):
    return train_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter(False, True))


def get_iterative_pred(model, modalities, masks, B, max_len, start_idx, pad_idx, end_idx, voc_size, device, greedy=False):
    with torch.no_grad():
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)
        probs = torch.zeros(B, 1).float().to(device)
        preds = torch.zeros(B,1,voc_size).float().to(device)
        segments = torch.zeros(B, 1).int().to(device)

        while (trg.size(-1) <= (max_len + 1)) and (not completeness_mask.all()):
            masks["C_mask"] = c_mask(trg, pad_idx)

            pred, worker_feat, manager_feat, goal_feat, segment_labels = model(modalities, trg, masks)#exploration here will not be exploration later?
            B,L,voc_size = pred.shape
            pred_probs = torch.exp(pred[:, -1])#TODO check

            if greedy:
                sampled_prediction = torch.max(pred_probs, dim=1)[1]
            else:
                dist = Categorical(pred_probs)
                sampled_prediction = dist.sample()

            sampled_probs = torch.gather(pred_probs, 1, sampled_prediction.unsqueeze(-1)).squeeze()

            trg = torch.cat([trg, sampled_prediction.unsqueeze(-1)], dim=-1)
            probs = torch.cat([probs, sampled_probs.unsqueeze(-1)], dim=-1)
            preds = torch.cat([preds, pred[:,-1].unsqueeze(1)], dim=1)
            segments = torch.cat([segments, segment_labels.reshape((B,L))[:,-1].unsqueeze(-1)], dim=-1)
            
            completeness_mask = completeness_mask | torch.eq(sampled_prediction, end_idx).byte()

        L = trg.shape[1]
        if L < max_len:
            test_print(f"maxlen {max_len}")
            pad_amnt = max_len - L
            trg = F.pad(trg, (0,pad_amnt), "constant", pad_idx)
            probs = F.pad(trg, (0,pad_amnt), "constant", 0)

    trg_input = trg[:, :-1]
    input_mask = c_mask(trg_input, pad_idx)
    return trg_input, input_mask, trg[:, 1:], probs[:,1:], preds[:,1:], segments[:,1:]



def train_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]
    cap_model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    device = get_device(cfg)

    if train_worker:
        wv_model.train()
        cap_model.module.teach_worker()
        reward_weight = cfg.rl_reward_weight_worker
        value_optimizer = wv_optimizer
        value_criterion = wv_criterion
    else:
        mv_model.train()
        cap_model.module.teach_manager()
        reward_weight = cfg.rl_reward_weight_manager
        value_optimizer = mv_optimizer
        value_criterion = mv_criterion

    progress_bar_name = f'{log_prefix} | {cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        cap_optimizer.zero_grad()
        value_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx, caption_idx_y, m1, masks = feature_getter(cfg, batch, loader)

        #with autograd.detect_anomaly():
            
        with autograd.detect_anomaly():
            try:
                prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model(m1, caption_idx, masks)

                loss_mask = (caption_idx_y != 1)

                token_mask = (caption_idx_y != loader.dataset.pad_idx)
                n_tokens = token_mask.sum()


                if train_worker:
                    expected_value = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
                else:
                    expected_value = mv_model((manager_feat.detach())).squeeze()

                losses, scores, samples, amplitude = sample_loss_kl(train_worker=train_worker, prediction=prediction, scorer=scorer, expected_scores=expected_value.detach(), trg=caption_idx_y, trg_caption=batch['captions'],
                    mask=loss_mask, segments = segment_labels, device=device, biased_kldiv=cap_criterion)

                cap_loss = torch.sum(losses) / n_tokens# if train_worker else torch.sum(losses)
                test_print(f'Loss: {cap_loss.item()}')
                cap_loss.backward()
                cap_optimizer.step()
                train_total_loss += cap_loss.item()
                    
            except Exception as e:
                test_print(str(e))

        #except Exception as e:
        #    test_print(str(e))

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(cap_model.parameters(), cfg.grad_clip)
        # -----------------------------------
        score = scores[0]
        # -----------value loss-------------
        loss_mask = loss_mask if train_worker else segment_labels.detach().float()
        #loss_mask *= score_mask #TODO experimental
        value_loss = value_criterion(expected_value, score) * loss_mask.float()
        value_loss = value_loss.mean()
        value_loss.backward()
        value_optimizer.step()

                #--------test logs ----------
        if (i % 100) == 0:
            log_iteration(loader, samples[0], caption_idx_y, score, expected_value, amplitude[0], segment_labels, train_worker)

            start_idx = loader.dataset.start_idx
            end_idx = loader.dataset.end_idx
            pad_idx = loader.dataset.pad_idx

            greedy = audio_decoder(cap_model.module, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality)#TODO variable video audio
            test_print(f'Greedy Decoder: {test_sentence(loader, greedy[0])}')

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(cap_optimizer), epoch)

def train_bmhrl_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker):
    return train_bimodal_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter(True, False))

def train_bimodal_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]

    cap_model.train()
    loader.dataset.update_iterator()
    train_total_loss = 0

    device = get_device(cfg)
    stabilize = cfg.rl_stabilize
    
    if train_worker:
        wv_model.train()
        cap_model.module.teach_worker()
        reward_weight = cfg.rl_reward_weight_worker
        value_optimizer = wv_optimizer
        value_criterion = wv_criterion
    else:
        mv_model.train()
        cap_model.module.teach_manager()
        reward_weight = cfg.rl_reward_weight_manager
        value_optimizer = mv_optimizer
        value_criterion = mv_criterion

    progress_bar_name = f'{log_prefix} | {cfg.curr_time[2:]}: train <{"W" if train_worker else "M"}>{epoch} @ {cfg.device}'

    #model_epoch = torch.tensor(epoch // 2).float()
    #model_factor = torch.sigmoid(model_epoch - 1).unsqueeze(-1)

    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx

    norm_factor = torch.tensor(20).float().to(device)
    impact_factor = torch.tensor(4).float().to(device)
    loss_factor = (impact_factor/norm_factor)

    #with autograd.detect_anomaly():

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        cap_optimizer.zero_grad()
        value_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx, caption_idx_y, (V,A), masks = feature_getter(cfg, batch, loader)
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)

        loss_mask = (caption_idx_y != loader.dataset.pad_idx)
        n_tokens = loss_mask.sum()

        if train_worker:
            expected_value = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
        else:
            expected_value = mv_model((manager_feat.detach())).squeeze()

        losses, scores, samples, amplitude = biased_kl(train_worker=train_worker, prediction=prediction, scorer=scorer, expected_scores=expected_value.detach(), trg=caption_idx_y, trg_caption=batch['captions'],
            mask=loss_mask, segments = segment_labels, device=device, biased_kldiv=cap_criterion, stabilize=stabilize)
            

        cap_loss = torch.sum(losses) / (n_tokens * loss_factor)
        test_print(f'Loss: {cap_loss.item()}')
        cap_loss.backward()
        cap_optimizer.step()
        train_total_loss += cap_loss.item()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(cap_model.parameters(), cfg.grad_clip)
        # -----------------------------------
        score = scores[0]
        # -----------value loss-------------
        loss_mask = loss_mask if train_worker else segment_labels.detach().float()
        value_loss = value_criterion(expected_value, score) * loss_mask.float()
        value_loss = value_loss.mean()
        value_loss.backward()
        value_optimizer.step()

        #--------test logs ----------
        if (i % 100) == 0:
            log_iteration(loader, samples[0], caption_idx_y, score, expected_value, amplitude[0], segment_labels, train_worker)
            greedy = bmhrl_greedy_decoder(cap_model.module, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality)
            test_print(f'Greedy Decoder: {test_sentence(loader, greedy[0])}')

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(cap_optimizer), epoch)

def analyze_bmhrl_div(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker):
    return analyze_bimodal_div(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter(True, False))


def print_example(loader, y, y_hat, y_hat_prob, biased_l, weighted_l, l, amplitude, score, reward, top_k=1):
    outliers = get_top_outliers(biased_l, l, top_k)
    index = outliers#[0]#TODO

    test_print("--"*25)
    test_print(f"GT:\t{y[index]}")
    test_print(f"HY:\t{test_sentence(loader, y_hat[index])}\n")
    test_print(f"Prob.:\t{y_hat_prob[index]}")
    test_print(f"Ampl.:\t{amplitude[index]}")
    test_print(f"Scr.:\t{score[index]}")
    test_print(f"Met.:\t{reward[index]}")
    test_print("--"*10)
    test_print(f"L:\t{l[index]}")
    test_print(f"BL:\t{biased_l[index]}")
    test_print(f"WL:\t{weighted_l[index]}")


def analyze_bimodal_div(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker, feature_getter):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]

    #kl_div = nn.KLDivLoss(reduction="none")

    cap_model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    device = get_device(cfg)
    #train_worker = False #---------------------------------------------------TODO REMOVE
    
    cap_model.module.teach_worker()
    
    progress_bar_name = f'{log_prefix} | {cfg.curr_time[2:]}: analyze {epoch} @ {cfg.device}'

    model_epoch = torch.tensor(epoch // 2).float()
    model_factor = torch.sigmoid(model_epoch - 1).unsqueeze(-1)

    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx

    norm_factor = torch.tensor(20).float().to(device)
    impact_factor = torch.tensor(4).float().to(device)
    loss_factor = (impact_factor/norm_factor)

    voc_size = loader.dataset.trg_voc_size
    kl_div = LabelSmoothing(0.7, pad_idx)#TODO inject

    with autograd.detect_anomaly():

        for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
            cap_optimizer.zero_grad()

            src = batch['feature_stacks']

            caption_idx, caption_idx_y, (V,A), masks = feature_getter(cfg, batch, loader)
            B,L = caption_idx.shape

            sampled_target, target_mask, sampled_target_y, sampled_probs, preds, segments = get_iterative_pred(cap_model, (V,A), masks, B, L-1, start_idx=start_idx, pad_idx=pad_idx, end_idx=end_idx, voc_size=voc_size,device=device,
            greedy=True)

            try:
                loss_mask = (caption_idx_y != loader.dataset.pad_idx)
                n_tokens = loss_mask.sum()

                test_print(test_sentence(loader, caption_idx_y[0]))

                losses, scores, samples = w_b_n_kl(train_worker=train_worker, prediction=preds, scorer=scorer, expected_scores=0, trg=caption_idx_y, trg_caption=batch['captions'],
                    mask=loss_mask, segments = segments, device=device, biased_kldiv=cap_criterion, kldiv=kl_div, norm_factor=norm_factor)
                
                w_l, b_l, l = losses
                amplitude, d_meteor, meteor = scores
                y_hat, y_hat_probs = samples

                print_example(loader, batch["captions"], y_hat, y_hat_probs, b_l, w_l, l, amplitude, d_meteor, meteor)

            except Exception as e:
                test_print(str(e))
                test_print(traceback.format_exc())
                continue


def warmstart_bmhrl_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard):
    return warmstart_bimodal_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, feature_getter(True, False))

def warmstart_bimodal_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, feature_getter):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]

    cap_model.train()#.cuda()
    wv_model.train()
    mv_model.train()
    loader.dataset.update_iterator()
    train_total_loss = 0
    progress_bar_name = f'{log_prefix} | {cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    cap_model.module.teach_warmstart()

    device = get_device(cfg)

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        cap_optimizer.zero_grad()
        mv_optimizer.zero_grad()
        wv_optimizer.zero_grad()

        caption_idx, caption_idx_y, (V,A), masks = feature_getter(cfg, batch, loader)

        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)
        token_mask = (caption_idx_y != loader.dataset.pad_idx)
        n_tokens = token_mask.sum()
        loss = torch.sum(cap_criterion(prediction, caption_idx_y)) / n_tokens
        loss.backward()
        cap_optimizer.step()

        with torch.no_grad():
            worker_score, manager_score, rewards = scorer.delta_meteor(torch.argmax(prediction, -1), batch['captions'], masks["C_mask"][:,-1], segment_labels)
            worker_score = worker_score.to(device)
            manager_score = manager_score.to(device)
        worker_loss_mask, manager_loss_mask = token_mask.float(), segment_labels.detach().float()

        #--------------wv warmstart-------------
        expected_worker_score = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
        worker_value_loss = wv_criterion(expected_worker_score, worker_score)
        worker_value_loss *= worker_loss_mask
        worker_value_loss.mean().backward()
        wv_optimizer.step()

        #--------------mv warmstart-------------
        expected_manager_score = mv_model((manager_feat.detach())).squeeze()
        manager_value_loss = mv_criterion(expected_manager_score, manager_score)
        manager_value_loss *= manager_loss_mask
        manager_value_loss.mean().backward()
        mv_optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(cap_optimizer), epoch)

def warmstart_audio_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard):
    warmstart_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, feature_getter(False, True))

def warmstart_video_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard):
    warmstart_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, feature_getter(False, False))

def warmstart_uni_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, feature_getter):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]

    cap_model.train()
    wv_model.train()
    mv_model.train()
    loader.dataset.update_iterator()
    train_total_loss = 0
    progress_bar_name = f'{log_prefix} | {cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    cap_model.module.teach_warmstart()

    device = get_device(cfg)

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        cap_optimizer.zero_grad()
        mv_optimizer.zero_grad()
        wv_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx, caption_idx_y, m1, masks = feature_getter(cfg, batch, loader)

        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model(m1, caption_idx, masks)
        prediction = prediction#TODO dont double log - careful
        token_mask = (caption_idx_y != loader.dataset.pad_idx)
        n_tokens = token_mask.sum()
        loss = torch.sum(cap_criterion(prediction, caption_idx_y)) / n_tokens
        loss.backward()
        cap_optimizer.step()

        c_mask = masks[1]

        with torch.no_grad():
            worker_score, manager_score = scorer.delta_meteor(torch.argmax(prediction, -1), batch['captions'], c_mask[:,-1], segment_labels)
            worker_score = worker_score.to(device)
            manager_score = manager_score.to(device)
        worker_loss_mask, manager_loss_mask = token_mask.float(), segment_labels.detach().float()

        #--------------wv warmstart-------------
        expected_worker_score = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
        worker_value_loss = wv_criterion(expected_worker_score, worker_score)
        worker_value_loss *= worker_loss_mask
        worker_value_loss.mean().backward()
        wv_optimizer.step()

        #--------------mv warmstart-------------
        expected_manager_score = mv_model((manager_feat.detach())).squeeze()
        manager_value_loss = mv_criterion(expected_manager_score, manager_score)
        manager_value_loss *= manager_loss_mask
        manager_value_loss.mean().backward()
        mv_optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(cap_optimizer), epoch)