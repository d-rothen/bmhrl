import numpy as np
# import tensorboardX as tensorboard
import torch
import os
from typing import Dict, List, Union

from torch.utils import tensorboard as tensorboard
from torch.utils.data import DataLoader
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from epoch_loops.captioning_bmrl_loops import bmhrl_greedy_decoder, bmhrl_inference, bmhrl_test, bmhrl_validation_next_word_loop, train_bmhrl, train_bmhrl_bl, warmstart_bmhrl, warmstart_bmhrl_2, warmstart_bmhrl_bl
from loss.rl_label_smoothing import RlLabelSmoothing
from metrics.batched_meteor import MeteorScorer
from model.bm_hrl_agent import BMHrlAgent, BMManagerValueFunction, BMWorkerValueFunction
from utilities.out_log import print_to_file as print_log

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


from captioning_datasets.captioning_dataset import ActivityNetCaptionsDataset
# from datasets.load_features import load_features_from_npy
from captioning_datasets.load_features import crop_a_segment, pad_segment
from epoch_loops.captioning_epoch_loops import make_masks
from model.captioning_module import BiModalTransformer
from model.proposal_generator import MultimodalProposalGenerator
from utilities.proposal_utils import (get_corner_coords,
                                      remove_very_short_segments,
                                      select_topk_predictions, trim_proposals, non_max_suppresion)


def get_learned_constants(model):
    p = lambda x: print(x, file=sys.stderr)
    for layer in model.module.bm_worker_fus.decoder.layers:
        p(f'Worker av: {layer.a_v_constant}')
    for layer in model.module.bm_manager_fus.decoder.layers:
        p(f'Manager av: {layer.a_v_constant}')


def get_bmhrl(cfg):
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
    #loader = DataLoader(dataset, collate_fn=dataset.dont_collate)


    #model = HRLAgent(cfg=cfg, vocabulary=train_dataset.train_vocab)
    model = BMHrlAgent(cfg, dataset)

    #if cfg.optimizer == 'adam':
    #    optimizer = torch.optim.Adam(model.parameters(), cfg.lr, (cfg.beta1, cfg.beta2), cfg.eps,
    #                                weight_decay=cfg.weight_decay)

    model.to(device)

    
    if cfg.rl_pretrained_model_dir is not None:
        print(f"Looking for pretrained model at {cfg.rl_pretrained_model_dir}", file=sys.stderr)
        loaded_model = model.load_model(cfg.rl_pretrained_model_dir)
        return model, dataset


def generate_proposals(
        prop_model: torch.nn.Module, feature_paths: Dict[str, str], pad_idx: int, cfg: Config, device: int,
        duration_in_secs: float
    ) -> torch.Tensor:
    '''Generates proposals using the pre-trained proposal model.

    Args:
        prop_model (torch.nn.Module): Pre-trained proposal model
        feature_paths (Dict): dict with paths to features ('audio', 'rgb', 'flow')
        pad_idx (int): A special padding token from train dataset.
        cfg (Config): config object used to train the proposal model
        device (int): GPU id
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        torch.Tensor: tensor of size (batch=1, num_props, 3) with predicted proposals.
    '''
    # load features
    feature_stacks = load_features_from_npy(
        feature_paths, None, None, duration_in_secs, pad_idx, device, get_full_feat=True,
        pad_feats_up_to=cfg.pad_feats_up_to
    )

    # form input batch
    batch = {
        'feature_stacks': feature_stacks,
        'duration_in_secs': duration_in_secs
    }

    with torch.no_grad():
        # masking out padding in the input features
        masks = make_masks(batch['feature_stacks'], None, cfg.modality, pad_idx)
        # inference call
        predictions, _, _, _ = prop_model(batch['feature_stacks'], None, masks)
        # (center, length) -> (start, end)
        predictions = get_corner_coords(predictions)
        # sanity-preserving clipping of the start & end points of a segment
        predictions = trim_proposals(predictions, batch['duration_in_secs'])
        # fildering out segments which has 0 or too short length (<0.2) to be a proposal
        predictions = remove_very_short_segments(predictions, shortest_segment_prior=0.2)
        # seƒlect top-[max_prop_per_vid] predictions
        predictions = select_topk_predictions(predictions, k=cfg.max_prop_per_vid)

    return predictions
    
    
def load_features_from_npy(
        feature_paths: Dict[str, str], start: float, end: float, duration: float, pad_idx: int,
        device: int, get_full_feat=False, pad_feats_up_to: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
    '''Loads the pre-extracted features from numpy files.
    This function is conceptually close to `datasets.load_feature.load_features_from_npy` but cleaned up
    for demonstration purpose.

    Args:
        feature_paths (Dict[str, str]): Paths to the numpy files (keys: 'audio', 'rgb', 'flow).
        start (float, None): Start point (in secs) of a proposal, if used for captioning the proposals.
        end (float, None): Ending point (in secs) of a proposal, if used for captioning the proposals.
        duration (float): Duration of the original video in seconds.
        pad_idx (int): The index of the padding token in the training vocabulary.
        device (int): GPU id.
        get_full_feat (bool, optional): Whether to output full, untrimmed, feature stacks. Defaults to False.
        pad_feats_up_to (Dict[str, int], optional): If get_full_feat, pad to this value. Different for audio
                                                    and video modalities. Defaults to None.

    Returns:
        Dict[str, torch.Tensor]: A dict holding 'audio', 'rgb' and 'flow' features.
    '''

    # load features. Please see README in the root folder for info on video features extraction
    stack_vggish = np.load(feature_paths['audio'])
    stack_rgb = np.load(feature_paths['rgb'])
    stack_flow = np.load(feature_paths['flow'])

    stack_vggish = torch.from_numpy(stack_vggish).float()
    stack_rgb = torch.from_numpy(stack_rgb).float()
    stack_flow = torch.from_numpy(stack_flow).float()

    # for proposal generation we pad the features
    if get_full_feat:
        stack_vggish = pad_segment(stack_vggish, pad_feats_up_to['audio'], pad_idx)
        stack_rgb = pad_segment(stack_rgb, pad_feats_up_to['video'], pad_idx)
        stack_flow = pad_segment(stack_flow, pad_feats_up_to['video'], pad_idx=0)
    # for captioning use trim the segment corresponding to a prop
    else:
        tmp_start = 0
        tmp_sec_end = 15
        tmp_end = 16
        stack_vggish = crop_a_segment(stack_vggish, tmp_start, tmp_sec_end, tmp_end)
        stack_rgb = crop_a_segment(stack_rgb, tmp_start, tmp_sec_end, tmp_end)
        stack_flow = crop_a_segment(stack_flow, tmp_start, tmp_sec_end, tmp_end)

    # add batch dimension, send to device
    stack_vggish = stack_vggish.to(torch.device(device)).unsqueeze(0)
    stack_rgb = stack_rgb.to(torch.device(device)).unsqueeze(0)
    stack_flow = stack_flow.to(torch.device(device)).unsqueeze(0)

    return {'audio': stack_vggish,'rgb': stack_rgb,'flow': stack_flow}


def caption_proposals(
        cap_model: torch.nn.Module, feature_paths: Dict[str, str],
        train_dataset: torch.utils.data.dataset.Dataset, cfg: Config, device: int, proposals: torch.Tensor,
        duration_in_secs: float
    ):
    '''Captions the proposals using the pre-trained model. You must specify the duration of the orignal video.

    Args:
        cap_model (torch.nn.Module): pre-trained caption model. Use load_cap_model() functions to obtain it.
        feature_paths (Dict[str, str]): dict with paths to features ('audio', 'rgb' and 'flow').
        train_dataset (torch.utils.data.dataset.Dataset): train dataset which is used as a vocab and for
                                                          specfial tokens.
        cfg (Config): config object which was used to train caption model. pre-trained model checkpoint has it
        device (int): GPU id to calculate on.
        proposals (torch.Tensor): tensor of size (batch=1, num_props, 3) with predicted proposals.
        duration_in_secs (float): duration of the video in seconds. Try this tool to obtain the duration:
            `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 in.mp4`

    Returns:
        List(Dict(str, Union(float, str))): A list of dicts where the keys are 'start', 'end', and 'sentence'.
    '''

    results = []

    with torch.no_grad():
        #for start, end, conf in proposals.squeeze():

        # load features
        feature_stacks = load_features_from_npy(
            feature_paths, 0, 100, duration_in_secs, train_dataset.pad_idx, device
        )

        # decode a caption for each segment one-by-one caption word
        ints_stack = bmhrl_greedy_decoder(
            cap_model, feature_stacks, cfg.max_len, train_dataset.start_idx, train_dataset.end_idx,
            train_dataset.pad_idx, cfg.modality
        )
        assert len(ints_stack) == 1, 'the func was cleaned to support only batch=1 (validation_1by1_loop)'

        # transform integers into strings
        strings = [train_dataset.train_vocab.itos[i] for i in ints_stack[0].cpu().numpy()]

        # remove starting token
        strings = strings[1:]
        # and remove everything after ending token
        # sometimes it is not in the list (when the caption is intended to be larger than cfg.max_len)
        try:
            first_entry_of_eos = strings.index('</s>')
            strings = strings[:first_entry_of_eos]
        except ValueError:
            pass

        # join everything together
        sentence = ' '.join(strings)
        # Capitalize the sentence
        sentence = sentence.capitalize()

        # add results to the list
        results.append({
            'start': 0,
            'end': 100,
            'sentence': sentence
        })

    return results

def load_prop_model(
        device: int, prop_generator_model_path: str, pretrained_cap_model_path: str, max_prop_per_vid: int
    ) -> tuple:
    '''Loading pre-trained proposal generator and config object which was used to train the model.

    Args:
        device (int): GPU id.
        prop_generator_model_path (str): Path to the pre-trained proposal generation model.
        pretrained_cap_model_path (str): Path to the pre-trained captioning module (prop generator uses the
                                         encoder weights).
        max_prop_per_vid (int): Maximum number of proposals per video.

    Returns:
        Config, torch.nn.Module: config, proposal generator
    '''
    # load and patch the config for user-defined arguments
    checkpoint = torch.load(prop_generator_model_path, map_location='cpu')
    cfg = checkpoint['config']
    cfg.device = device
    cfg.max_prop_per_vid = max_prop_per_vid
    cfg.pretrained_cap_model_path = pretrained_cap_model_path
    cfg.train_meta_path = './data/train.csv'  # in the saved config it is named differently

    # load anchors
    anchors = {
        'audio': checkpoint['anchors']['audio'],
        'video': checkpoint['anchors']['video']
    }

    # define model and load the weights
    model = MultimodalProposalGenerator(cfg, anchors)
    device = torch.device(cfg.device)
    torch.cuda.set_device(device)
    model.load_state_dict(checkpoint['model_state_dict'])  # if IncompatibleKeys - ignore
    model = model.to(cfg.device)
    model.eval()

    return cfg, model

def predict(cfg):
    #model = HRLAgent(cfg=cfg, vocabulary=train_dataset.train_vocab)
    model, dataset = get_bmhrl(cfg)
    vid_name = "in"#"shahin_judo"
    feature_paths = {
        'audio': f"/home/rothenda/video_features/output/{vid_name}_vggish.npy",
        'rgb': f"/home/rothenda/video_features/output/{vid_name}_rgb.npy",
        'flow': f"/home/rothenda/video_features/output/{vid_name}_flow.npy"
    }

    pred = caption_proposals(model, feature_paths, dataset, cfg, 0, proposals=None, duration_in_secs=None)
    print(pred[0], file=sys.stderr)

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