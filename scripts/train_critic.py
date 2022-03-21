import torch
from torchtext import data
import spacy
import numpy as np


def caption_iterator(cfg, batch_size, phase="init", override_file_path=""):
    print(f'Contructing caption_iterator for "{phase}" phase')
    spacy_en = spacy.load('en_core_web_sm')
    
    filepath = override_file_path if len(override_file_path) > 0 else cfg.train_csv_path

    def tokenize_en(txt):
        return [token.text for token in spacy_en.tokenizer(txt)]
    
    def tokenize_labels(labels):
        #return [F.one_hot(torch.tensor(int(float(label))), 2) for label in labels.split()]
        return np.array([int(float(label)) for label in labels.split()])
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', tokenizer_language="en_core_web_sm", init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    SEG_LABELS = data.ReversibleField(
        tokenize=tokenize_labels, init_token=cfg.segment_init_token, eos_token=cfg.segment_padding_index, 
        pad_token=cfg.segment_padding_index, batch_first=True, use_vocab=False
    )
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('seg_labels', SEG_LABELS),
        ('phase', None),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.vocab_csv_path, format='tsv', skip_header=True, fields=fields,
    )
    
    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    train_vocab = CAPTION.vocab
    
    dataset = data.TabularDataset(
        path=filepath, format='tsv', skip_header=True, fields=fields,
    )
    
    if phase == 'val_1':
        dataset = data.TabularDataset(path=cfg.val_1_meta_path, format='tsv', skip_header=True, fields=fields)
    elif phase == 'val_2':
        dataset = data.TabularDataset(path=cfg.val_2_meta_path, format='tsv', skip_header=True, fields=fields)
    elif phase == 'learned_props':
        dataset = data.TabularDataset(path=cfg.val_prop_meta_path, format='tsv', skip_header=True, fields=fields)

    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
    datasetloader = data.BucketIterator(dataset, batch_size, sort_key=lambda x: 0, 
                                        device=torch.device(cfg.device), repeat=False, shuffle=True)
    return train_vocab, datasetloader

def train_critic(cfg):
    segmentation_model = SegmentationModule(cfg)

    return