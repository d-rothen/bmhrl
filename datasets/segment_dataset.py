import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchtext import data

from datasets.load_features import fill_missing_features, load_features_from_npy

ds_path = '../data/CharadeCaptions'

class SegmentDataset(Dataset):
    def __init__(cfg):
        self.dataset = pd.read_json(cfg.train_segment_json_path)
        flat_captions = []
        flat_segments = []
        
        start_token = cfg.start_token
        end_token = cfg.end_token
        pad_token = cfg.pad_token
        max_pred_len = cfg.max_len

        for entry in self.dataset.iterrows():
            item = entry[1]
            captions = item['captions']
            segments = item['seg_labels']
            
            for i in range(len(segments)):
                og_caption = captions[i]
                
                #TODO handle ,
                caption = og_caption.split()#.replace('.', '').split()
                caption[-1] = end_token
                words = [start_token, *caption]#, eos]

                segment_labels = [0, *segments[i]]
                
                l_w, l_l = len(words), len(segment_labels)
                assert l_w == l_l, f'{og_caption} {l_w}, {str(segment_labels)} {l_l}'
                
                flat_captions.append(words)
                flat_segments.append(segment_labels)
                
        self.dataset = pd.DataFrame({
            'captions': flat_captions,
            'segments': flat_segments
        })
            
    def __getitem__(self, indices):
        captions = []
        segments = []
        
        for idx in indices:
            c,s = self.dataset.iloc[idx]
            captions.append(c)
            segments.append(s)
        return captions, segments
        #return {'captions': captions, 'segments': segments}

    def __len__(self):
        return len(self.dataset)

