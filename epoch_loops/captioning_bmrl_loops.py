from email.utils import parsedate_to_datetime
import sys
from tqdm import tqdm
import torch
from torch.distributions.categorical import Categorical

from epoch_loops.captioning_epoch_loops import make_masks
from metrics.batched_meteor import MeteorScorer
from utilities.captioning_utils import get_lr
import torch.nn as nn
import torch.nn.functional as F
from scripts.device import get_device
from torch import autograd

def bmhrl_inference(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality, captions, batch):
    src = feature_stacks

    caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
    caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
    masks = make_masks(batch['feature_stacks'], caption_idx, modality, pad_idx)

    V, A = src['rgb'] + src['flow'], src['audio']

    prediction = model.module.inference((V,A), caption_idx, masks)

    predicted_words = torch.max(prediction, -1)

    return predicted_words[1]#Tuple -> return indices
    
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

        test_1 = bmhrl_inference(cap_model, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality, None, batch)
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

def normalize_reward(reward):
    reward -= torch.mean(reward)
    reward /= torch.std(reward)
    return reward

def rl_loss(log_l, reward, mask):
    B,L = mask.shape
    log_l *= reward #TODO High reward -> should account for whole distribution
    _,Ll = log_l.shape
    zero_padding = Ll - L
    padded_loss_mask = torch.nn.functional.pad(mask, (0,zero_padding,0,0), value=0)
    
    log_l = (log_l * padded_loss_mask)
    log_l[log_l != log_l] = 0#trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

    #------normalize reward----------

    loss = torch.sum(log_l, dim=-1)
    loss = -torch.mean(loss)

    return loss

def gradient_analysis(cfg, models, scorer, loader, epoch, TBoard, train_worker):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]

    cap_model.eval()#.cuda()
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

    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):

        #Grad 1
        cap_optimizer.zero_grad()
        value_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)

        ##--------temp old------
        log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()
        loss_mask = (caption_idx_y != 1).float()#TODO use preset token for padding

        B,L = loss_mask.shape

        _,Ll = log_l.shape
        zero_padding = Ll - L
        padded_loss_mask = torch.nn.functional.pad(loss_mask, (0,zero_padding,0,0), value=0)
        
        log_l = (log_l * padded_loss_mask)
        log_l[log_l != log_l] = 0#trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

        loss = -torch.sum(log_l)
        loss.backward()
        grad1 = torch.clone(cap_model.module.worker.projection.weight.grad)
        #cap_optimizer.step()

        # ---------- Control ------------ 

        cap_optimizer.zero_grad()
        value_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)

        ##--------temp old------
        log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()
        loss_mask = (caption_idx_y != 1).float()#TODO use preset token for padding

        B,L = loss_mask.shape

        _,Ll = log_l.shape
        zero_padding = Ll - L
        padded_loss_mask = torch.nn.functional.pad(loss_mask, (0,zero_padding,0,0), value=0)
        
        log_l = (log_l * padded_loss_mask)
        log_l[log_l != log_l] = 0#trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

        loss = -torch.sum(log_l)
        loss.backward()
        grad_control = torch.clone(cap_model.module.worker.projection.weight.grad)
        pass

        #Grad 2
        ##----------------------

        cap_optimizer.zero_grad()
        value_optimizer.zero_grad()

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)

        with torch.no_grad():
            predicted_tokens = torch.argmax(prediction, -1)
            if train_worker:
                score = scorer.delta_meteor_worker(predicted_tokens, batch['captions'], masks["C_mask"][:,-1])
            else:
                score = scorer.delta_meteor_manager(predicted_tokens, batch['captions'], masks["C_mask"][:,-1], segment_labels)
            score_mask = (score != 0).float()
        if train_worker:
            expected_value = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
        else:
            expected_value = mv_model((manager_feat.detach())).squeeze()

        score = score.to(device)

        # ------------cap loss--------------
        expected_value_baseline = expected_value.detach() * score_mask#TODO this is experimental in order to not just flatten the whole vocab
        return_value = score - expected_value_baseline
        scaled_return_value = return_value * reward_weight

        log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()
        loss_mask = (caption_idx_y != 1).float()#TODO use preset token for padding
        B,L = loss_mask.shape

        _,Ll = log_l.shape
        zero_padding = Ll - L
        padded_loss_mask = torch.nn.functional.pad(loss_mask, (0,zero_padding,0,0), value=0)
        
        log_l = (log_l * padded_loss_mask)
        log_l[log_l != log_l] = 0#trick to remove nans, that appeared by multiplying 0 times -inf (-inf from log operation on 0 values)

        log_l *= scaled_return_value
        loss = -torch.sum(log_l)
        loss.backward()
        grad2 = torch.clone(cap_model.module.worker.projection.weight.grad)
        pass

def apply_ce_loss(loss, prediction, target):
    B,L,D = prediction.shape
    reshaped_pred = torch.reshape(prediction, (B,D,L))
    return loss(reshaped_pred, target)

def wang_loss(prediction, target, reward, device):
    B,L = target.shape
    target = target.unsqueeze(-1)
    #vals = torch.gather(prediction, 2, target)
    neg = torch.ones(B,L,1).to(device) * -1
    loss = torch.scatter_add(prediction, 2, target, neg)
    loss = torch.sum(prediction, dim=-1)
    loss = torch.log(loss)
    loss *= reward
    #loss[loss != loss] = 0#Remove potential NANs
    return loss

def wang_loss_rl(prediction):
    probs = Categorical(prediction)
    samples = probs.sample()
    log_probs = -probs.log_prob(samples)
    #vals = torch.gather(prediction, 2, target)
    log_probs[log_probs != log_probs] = 0#Remove potential NANs
    return log_probs

def get_score(train_worker, scorer, predicted_tokens, caption, mask, segments):
    if train_worker:
        return scorer.delta_meteor_worker(predicted_tokens, caption, mask)
    return scorer.delta_meteor_manager(predicted_tokens, caption, mask, segments)

def normed_kl(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, kl_div, pad_idx):
    B,S,V = prediction.shape
    smoothing = 0.7
    trg_factor = 1 - smoothing

    #diagnose
    nan_check = prediction[prediction != prediction]
    if nan_check.shape != torch.tensor([]).shape:
        test_print(prediction)
    #-----

    dist = Categorical(prediction)
    sampled = dist.sample()#TODO this samples on log probs not probs

    probs = torch.gather(prediction, 2, sampled.unsqueeze(-1)).squeeze()

    score = get_score(train_worker, scorer, sampled, trg_caption, mask, segments).to(device)
    unexpected_score = torch.clamp(score - expected_scores, -1,1)

    norm_reward_factor = torch.sum(mask, dim=-1)
    norm_reward_factor = torch.reshape(norm_reward_factor, (-1,1))

    amplitude = unexpected_score * norm_reward_factor.float() * trg_factor * probs

    base_amplitude = 1/(V-2)#TODO evaluate
    biased_ampl = amplitude + base_amplitude
    biased_ampl = torch.clamp(biased_ampl, 0, 1)

    test_print(torch.max(biased_ampl))
    test_print(torch.mean(biased_ampl))

    trg_ampl = (trg_factor - biased_ampl).contiguous().view(-1)

    biased_dist = torch.zeros_like(prediction)
    biased_dist = torch.scatter(biased_dist, 2, sampled.unsqueeze(-1), biased_ampl.unsqueeze(-1))

    # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
    pred = prediction.contiguous().view(-1, V)
    target = trg.contiguous().view(-1)
    
    # prior (uniform)
    dist = smoothing * torch.ones_like(pred) / (V - 2)
    # add smoothed ground-truth to prior (args: dim, index, src (value))
    dist.scatter_(1, target.unsqueeze(-1).long(), trg_ampl.unsqueeze(-1)) #Essentially "One Hot" encode traget with .3 (rest is 1/vocsize-1 * .7)
    # make the padding token to have zero probability
    dist[:, pad_idx] = 0
    dist = dist + biased_dist.contiguous().view(-1, V)
    # ?? mask: 1 if target == pad_idx; 0 otherwise 
    mask = torch.nonzero(target == pad_idx)
    
    if mask.sum() > 0 and len(mask) > 0: #(padded sentences are present)
        # dim, index, val
        dist.index_fill_(0, mask.squeeze(), 0) #set distance 0 where there are padding tokens

    divergence = kl_div(pred, dist)

    return divergence, [score], [sampled], [amplitude]
    

def sample_loss_kl(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, kl_div, pad_idx):
    B, S, V = prediction.shape
    smoothing = 0.7
    trg_factor = 1 - smoothing


    pred_probs = torch.exp(prediction)
    dist = Categorical(pred_probs)
    greedy_pred = dist.sample()

    #greedy_pred = torch.argmax(prediction, -1)

    #dist = Categorical(prediction)
    #greedy_pred = dist.sample()


    score = get_score(train_worker, scorer, greedy_pred, trg_caption, mask, segments).to(device)

    if not train_worker:
        log_probs = torch.gather(prediction, 2, greedy_pred.unsqueeze(-1)).squeeze()
        loss = -log_probs * score
        return loss, [score], [greedy_pred], [0]

    log_probs = torch.gather(pred_probs, 2, greedy_pred.unsqueeze(-1)).squeeze()

    test_print(f"\nProbs. : min = {torch.min(log_probs)}, max = {torch.max(log_probs)}")



    #dist_factor = F.sigmoid((score * torch.exp(log_probs)) * 3)#TODO just testing
    #biased_ampl = dist_factor * trg_factor

    norm_reward_factor = torch.sum(mask, dim=-1)
    norm_reward_factor = torch.reshape(norm_reward_factor, (-1,1))

    test_print(f"NormFac. : min = {torch.min(norm_reward_factor)}, max = {torch.max(norm_reward_factor)}") 


    score = torch.clamp(score, 0, 1)
    amplitude = score * trg_factor * log_probs * norm_reward_factor.float() 
    #amplitude = torch.clamp(amplitude, 0, 1)#TODO make use of negative rewards

    biased_ampl = amplitude

    test_print(f"Amplitude : min = {torch.min(biased_ampl)}, mean = {torch.mean(biased_ampl)}, max = {torch.max(biased_ampl)}") 

    #test_print(biased_ampl)

    trg_ampl = (trg_factor - biased_ampl).contiguous().view(-1)

    #torch.scatter

    biased_dist = torch.zeros_like(prediction)
    biased_dist = torch.scatter(biased_dist, 2, greedy_pred.unsqueeze(-1), biased_ampl.unsqueeze(-1))


    # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
    pred = prediction.contiguous().view(-1, V)
    target = trg.contiguous().view(-1)
    
    # prior (uniform)
    dist = smoothing * torch.ones_like(pred) / (V - 2)
    # add smoothed ground-truth to prior (args: dim, index, src (value))
    dist.scatter_(1, target.unsqueeze(-1).long(), trg_ampl.unsqueeze(-1)) #Essentially "One Hot" encode traget with .3 (rest is 1/vocsize-1 * .7)
    # make the padding token to have zero probability
    dist[:, pad_idx] = 0
    dist = dist + biased_dist.contiguous().view(-1, V)
    # ?? mask: 1 if target == pad_idx; 0 otherwise 
    mask = torch.nonzero(target == pad_idx)
    
    if mask.sum() > 0 and len(mask) > 0: #(padded sentences are present)
        # dim, index, val
        dist.index_fill_(0, mask.squeeze(), 0) #set distance 0 where there are padding tokens

    #test_print(pred[1])
    #test_print(dist[1])

    divergence = kl_div(pred, dist)

    #test_print(divergence)

    test_print(f"Divergence. : min = {torch.min(divergence)}, max = {torch.max(divergence)}") 

    return divergence, [score], [greedy_pred], [amplitude]

def sample_loss_kl_old(train_worker, prediction, scorer, expected_scores, trg, trg_caption, mask, segments, device, kl_div, pad_idx):
    B, S, V = prediction.shape
    smoothing = 0.7
    trg_factor = 1 - smoothing

    greedy_pred = torch.argmax(prediction, -1)
    log_probs = torch.gather(prediction, 2, greedy_pred.unsqueeze(-1)).squeeze()
    score = get_score(train_worker, scorer, greedy_pred, trg_caption, mask, segments).to(device)
    #dist_factor = F.sigmoid((score * torch.exp(log_probs)) * 3)#TODO just testing
    #biased_ampl = dist_factor * trg_factor

    test_print(torch.min(score))
    test_print(torch.max(score))
    test_print(f'NaNs: {torch.sum(score != score)}')


    norm_reward_factor = torch.sum(mask, dim=-1)
    norm_reward_factor = torch.reshape(norm_reward_factor, (-1,1))
    biased_ampl = torch.clamp(score, 0,1) * torch.exp(log_probs) * trg_factor# * norm_reward_factor.float()


    

    test_print(torch.max(biased_ampl))
    test_print(torch.mean(biased_ampl))

    #test_print(biased_ampl)

    trg_ampl = (trg_factor - biased_ampl).contiguous().view(-1)

    #torch.scatter

    biased_dist = torch.zeros_like(prediction)
    biased_dist = torch.scatter(biased_dist, 2, greedy_pred.unsqueeze(-1), biased_ampl.unsqueeze(-1))


    # (B, S, V) -> (B * S, V); (B, S) -> (B * S)
    pred = prediction.contiguous().view(-1, V)
    target = trg.contiguous().view(-1)
    
    # prior (uniform)
    dist = smoothing * torch.ones_like(pred) / (V - 2)
    # add smoothed ground-truth to prior (args: dim, index, src (value))
    dist.scatter_(1, target.unsqueeze(-1).long(), trg_ampl.unsqueeze(-1)) #Essentially "One Hot" encode traget with .3 (rest is 1/vocsize-1 * .7)
    # make the padding token to have zero probability
    dist[:, pad_idx] = 0
    dist = dist + biased_dist.contiguous().view(-1, V)
    # ?? mask: 1 if target == pad_idx; 0 otherwise 
    mask = torch.nonzero(target == pad_idx)
    
    if mask.sum() > 0 and len(mask) > 0: #(padded sentences are present)
        # dim, index, val
        dist.index_fill_(0, mask.squeeze(), 0) #set distance 0 where there are padding tokens

    divergence = kl_div(pred, dist)

    return divergence, [score], [greedy_pred], [biased_ampl]



def sample_loss(train_worker, prediction, scorer, expected_scores, trg, trg_caption, prediction_mask, segments, n_samples, device):
    greedy_pred = torch.argmax(prediction, -1)
    log_probs = torch.gather(prediction, 2, greedy_pred.unsqueeze(-1)).squeeze()

    score = get_score(train_worker, scorer, greedy_pred, trg_caption, prediction_mask, segments).to(device)
    score.requires_grad_(True)

    test_loss = -(log_probs * score * prediction_mask.float())

    return test_loss, [score], [greedy_pred]


    B,L,V = prediction.shape
    pred_distribution = Categorical(prediction)
    samples = pred_distribution.sample_n(n_samples)

    with torch.no_grad():
        scores = torch.zeros(n_samples, B, L).to(device)
        for n in range(n_samples):
            scores[n] = get_score(train_worker, scorer, samples[n], trg_caption, prediction_mask, segments).to(device)

    factor = scores[0]
    factor.requires_grad_(True)
    test_loss = -(log_probs * scores[0] * prediction_mask.float())

    return test_loss, scores, [greedy_pred]
    #sample_returns = scores - expected_scores.unsqueeze(0).detach()
    sample_returns = scores

    losses = torch.zeros(n_samples, B, L).to(device)
    for n in range(n_samples):
        log_probs = pred_distribution.log_prob(samples[n])
        log_probs *= prediction_mask.float()
        #log_probs *= sample_returns[n]
        losses[n] = -log_probs
    
    return losses, scores, samples

def log_iteration(loader, pred, trg, score, score_pred, amplitude, segments, train_worker):
    B,L = pred.shape
    test_print(f'Summed Score: {torch.sum(score)}')

    for b in range(B):
        test_print(f'Pred[{b}]: {test_sentence(loader, pred[b])}')
        test_print(f'Trg[{b}]: {test_sentence(loader, trg[b])}')#TODO this doesnt show up?
        test_print(f'Score[{b}]: {score[b]}')
        test_print(f'Score_pred[{b}]: {score_pred[b]}')
        if train_worker:
            test_print(f'Ampl[{b}]: {amplitude[b]}')
        test_print(f'Segm[{b}]: {segments[b]}')


def train_bmhrl_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard, train_worker):
    cap_model, cap_optimizer, cap_criterion = models["captioning"]
    wv_model, wv_optimizer, wv_criterion = models["worker"]
    mv_model, mv_optimizer, mv_criterion = models["manager"]

    kl_div = nn.KLDivLoss(reduction="none")

    cap_model.train()#.cuda()
    loader.dataset.update_iterator()
    train_total_loss = 0

    device = get_device(cfg)
    #train_worker = False #---------------------------------------------------TODO REMOVE
    
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

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']

        #with autograd.detect_anomaly():
            
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)

        loss_mask = (caption_idx_y != 1)

        if train_worker:
            expected_value = wv_model((worker_feat.detach(), goal_feat.detach())).squeeze()
        else:
            expected_value = mv_model((manager_feat.detach())).squeeze()


        token_mask = (caption_idx_y != loader.dataset.pad_idx)
        n_tokens = token_mask.sum()

        #try:
        losses, scores, samples, amplitude = sample_loss_kl(train_worker=train_worker, prediction=prediction, scorer=scorer, expected_scores=expected_value.detach(), trg=caption_idx_y, trg_caption=batch['captions'],
            mask=loss_mask, segments = segment_labels, device=device, pad_idx=loader.dataset.pad_idx, kl_div=kl_div)

        #losses, scores, samples, amplitude = normed_kl(train_worker=train_worker, prediction=prediction, scorer=scorer, expected_scores=expected_value.detach(), trg=caption_idx_y, trg_caption=batch['captions'],
        #   mask=loss_mask, segments = segment_labels, device=device, pad_idx=loader.dataset.pad_idx, kl_div=kl_div)
    
        #log_l = torch.gather(prediction, 2, torch.unsqueeze(caption_idx_y,-1)).squeeze()

        # ------------cap loss--------------
        #cap_loss = torch.sum(losses, -1)
        #cap_loss = torch.mean(cap_loss, -1)
        #cap_loss = torch.mean(cap_loss, -1)
        cap_loss = torch.sum(losses) / n_tokens# if train_worker else torch.sum(losses)
        test_print(f'Loss: {cap_loss.item()}')
        cap_loss.backward()
        cap_optimizer.step()
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

        train_total_loss += cap_loss.item()


        #--------test logs ----------
        if (i % 100) == 0:
            log_iteration(loader, samples[0], caption_idx_y, score, expected_value, amplitude[0], segment_labels, train_worker)

            start_idx = loader.dataset.start_idx
            end_idx = loader.dataset.end_idx
            pad_idx = loader.dataset.pad_idx

            greedy = bmhrl_greedy_decoder(cap_model.module, src, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality)
            test_print(f'Greedy Decoder: {test_sentence(loader, greedy[0])}')

    train_total_loss_norm = train_total_loss / len(loader)
    
    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(cap_optimizer), epoch)

def warmstart_bmhrl_bl(cfg, models, scorer, loader, epoch, log_prefix, TBoard):
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
        
        #caption_idx = batch['caption_data'].caption 
        #caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        #masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)#video and audio feature vectors are 1024/128 1es for "non processed" video/audio -> create bool mask

        src = batch['feature_stacks']

        caption_idx = batch['caption_data'].caption #batchsize x max_seq_len, vocab indices
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        V, A = src['rgb'] + src['flow'], src['audio']
        prediction, worker_feat, manager_feat, goal_feat, segment_labels = cap_model((V,A), caption_idx, masks)
        prediction = torch.log(prediction)#TODO dont double log - careful
        token_mask = (caption_idx_y != loader.dataset.pad_idx)
        n_tokens = token_mask.sum()
        loss = torch.sum(cap_criterion(prediction, caption_idx_y)) / n_tokens
        loss.backward()
        cap_optimizer.step()

        with torch.no_grad():
            worker_score, manager_score = scorer.delta_meteor(torch.argmax(prediction, -1), batch['captions'], masks["C_mask"][:,-1], segment_labels)
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