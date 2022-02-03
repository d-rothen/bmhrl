import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchtext import data
import numpy as np
import spacy

class SegmentDataset(Dataset):
    def __init__(self):
        self.dataset = pd.read_json(ds_path)
        flat_captions = []
        flat_segments = []
        
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


def caption_iterator(cfg):
    spacy_en = spacy.load('en')
    
    def tokenize_en(txt):
        return [token.text for token in spacy_en.tokenizer(txt)]
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    
    # the order has to be the same as in the table
    fields = [
        ('caption', CAPTION)
    ]

    dataset = data.TabularDataset(
        path=cfg.segmentation_vocab_path, format='tsv', skip_header=True, fields=fields,
    )
    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    train_vocab = CAPTION.vocab

    return train_vocab


def combined_vocab(cfg):
    #spacy_en = spacy.load('en_core_web_sm')
    CAPTION = data.ReversibleField(
        tokenize='spacy', init_token=cfg.start_token, eos_token=cfg.end_token, # tokenizer_language='en_core_web_sm',
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )

    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )

    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('idx', INDEX),
    ]

    dataset = data.TabularDataset(
        path=cfg.segmentation_vocab_path, format='tsv', fields=fields, skip_header=True
    )

    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)

    train_vocab = CAPTION.vocab
    
    return train_vocab

def train_vocab(cfg):
    spacy_en = spacy.load('en_core_web_sm')
    
    def tokenize_en(txt):
        return [token.text for token in spacy_en.tokenizer(txt)]
    
    CAPTION = data.ReversibleField(
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )

    CAPTION_CHARADES = data.ReversibleField(
        tokenize='spacy', tokenizer_language='en_core_web_sm', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    
    INDEX = data.Field(
        sequential=False, use_vocab=False, batch_first=True
    )
    
    fields = [
        ('video_id', None),
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('phase', None),
        ('idx', INDEX),
    ]

    fields_charades = [
        ('id', None),
        ('captions', CAPTION),
        ('seg_labels', None),
        ('scene', None),
        ('actions', None),
        ('length', None),
        ('path', None),
        ('split', None)
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    )

    dataset_charades = data.TabularDataset(
        path=cfg.train_segmentation_json, format='json', fields = fields_charades
    )

    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    CAPTION_CHARADES.build_vocab(dataset_charades.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps)
    train_vocab = CAPTION.vocab
    
    return train_vocab

class VocabularyEmbedder(nn.Module):

    def __init__(self, voc_size, emb_dim):
        super(VocabularyEmbedder, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        # replaced if pretrained weights are used
        self.embedder = nn.Embedding(voc_size, emb_dim)

    def forward(self, x):  # x - tokens (B, seq_len)
        x = self.embedder(x)
        x = x * np.sqrt(self.emb_dim)

        return x  # (B, seq_len, d_model)

    def init_word_embeddings(self, weight_matrix, emb_weights_req_grad=True):
        if weight_matrix is None:
            print('Training word embeddings from scratch')
        else:
            pretrained_voc_size, pretrained_emb_dim = weight_matrix.shape
            if self.emb_dim == pretrained_emb_dim:
                self.embedder = self.embedder.from_pretrained(weight_matrix)
                self.embedder.weight.requires_grad = emb_weights_req_grad
                print('Glove emb of the same size as d_model_caps')
            else:
                self.embedder = nn.Sequential(
                    nn.Embedding(self.voc_size, pretrained_emb_dim).from_pretrained(weight_matrix),
                    nn.Linear(pretrained_emb_dim, self.emb_dim),
                    nn.ReLU()
                )
                self.embedder[0].weight.requires_grad = emb_weights_req_grad

class Critic(nn.Module):
    def __init__(self, hidden_size, d_voc_size):
        super(Critic, self)
        #TODO do embedding in forward or pass embedding?
        word_embedding_size = d_voc_size
        # input word sequence to determine if end of segment reached
        #Batch first so batch is first dimension for fc

        num_layers = 2
        p_dout = 0.1

        self.gru = nn.GRU(input_size=word_embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=p_dout)
        self.rnn = nn.RNN(input_size=word_embedding_size, hidden_size=hidden_size, batch_first=True)
        #TODO classification für [continue, end] oder nur [end] - paper lässt auf [end] schließen
        self.fc = nn.Linear(hidden_size, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, segment):
        output, h_n = self.rnn(segment)
        return self.sigmoid(output)[:,:,0]

class SegmentationModule(nn.Module):
    def __init__(self, cfg) -> None:
        super(SegmentationModule, self).__init__()
        #vocab = caption_iterator(cfg)
        #vocab_size = len(vocab)
        #vocab_vectors = vocab.vectors
        #voc_embedder = VocabularyEmbedder(vocab_size, cfg.d_model_caps)


