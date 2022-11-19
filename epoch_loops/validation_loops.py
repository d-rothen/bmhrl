import os
import json
from tqdm import tqdm
import torch
import spacy
from time import time

from evaluation.evaluate import ANETcaptions
from model.masking import mask, make_masks
from scripts.device import get_device
from utilities.captioning_utils import HiddenPrints, get_lr

def validation_1by1_loop(cfg, model, loader, decoder, epoch, TBoard):
    start_timer = time()
    
    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True, 
            'details': ''
        },
        'results': {}
    }
    model.eval()
    loader.dataset.update_iterator()
    
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names
    
    if phase == 'val_1':
        reference_paths = [cfg.reference_paths[0]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'val_2':
        reference_paths = [cfg.reference_paths[1]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'learned_props':
        reference_paths = cfg.reference_paths  # here we use all of them
        tIoUs = cfg.tIoUs
        assert len(tIoUs) == 4

    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} 1by1 {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        # caption_idx = batch['caption_data'].caption
        # caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        ints_stack = decoder(
            model.module, batch['feature_stacks'], cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality#, batch["captions"], batch
        )
        ints_stack = ints_stack.cpu().numpy()  # what happens here if I use only cpu?
        # transform integers into strings
        list_of_lists_with_strings = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack]
        ### FILTER PREDICTED TOKENS
        # initialize the list to fill it using indices instead of appending them
        list_of_lists_with_filtered_sentences = [None] * len(list_of_lists_with_strings)

        for b, strings in enumerate(list_of_lists_with_strings):
            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()
            # add the filtered sentense to the list
            list_of_lists_with_filtered_sentences[b] = sentence
            
        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sent in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                              list_of_lists_with_filtered_sentences):
            segment = {
                'sentence': sent,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)

            else:
                predictions['results'][video_id] = [segment]
    
    if cfg.log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        save_filename = f'captioning_results_{phase}_e{epoch}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # if this is run with another loader and pretrained model
        # it substitutes the previous prediction
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        ## RUN THE EVALUATION
        # blocks the printing
        with HiddenPrints():
            val_metrics = calculate_metrics(reference_paths, submission_path, tIoUs, cfg.max_prop_per_vid)

        if phase == 'learned_props':
            print(submission_path)

        ## WRITE TBOARD
        if (TBoard is not None) and (phase != 'learned_props'):
            # todo: add info that this metrics are calculated on val_1
            TBoard.add_scalar(f'{phase}/meteor', val_metrics['Average across tIoUs']['METEOR'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu4', val_metrics['Average across tIoUs']['Bleu_4'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu3', val_metrics['Average across tIoUs']['Bleu_3'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/precision', val_metrics['Average across tIoUs']['Precision'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/recall', val_metrics['Average across tIoUs']['Recall'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/duration_of_1by1', (time() - start_timer) / 60, epoch)

        return val_metrics

def validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        masks = make_masks(batch['feature_stacks'], caption_idx, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model(batch['feature_stacks'], caption_idx, masks)
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(pred, caption_idx_y) / n_tokens
            val_total_loss += loss.item()
            
    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm

def calculate_metrics(reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose=True, only_proposals=False):
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