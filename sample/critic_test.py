from captioning_datasets.captioning_dataset import caption_iterator
from model.hrl_agent import HRLAgent
import sys
import torch
import numpy as np
from pycocoevalcap.meteor.meteor import Meteor

scorer = Meteor()

test_sentences = [
    "there is a man standing in the rain",
    "he is holdin an umbrella <blank> <blank> <blank>",
    "there are other people around <blank> <blank> <blank>",
    "a bird flies on top of his head",
    "the man hits the bird with his hand",
    "the bird flies away screeching <blank> <blank> <blank>",
    "a taxi arrives next to a man <blank>",
    "a man enters a taxi <blank> <blank> <blank>"
]

def run_critic_test(cfg):
    batch_size=32
    train_vocab, caption_loader = caption_iterator(cfg, batch_size, "train")

    s = scorer.compute_score([["Hello there"]], [["Hello there <blank>"]])
    print("score: " + str(s))

    model = HRLAgent(cfg=cfg, vocabulary=train_vocab)
    print(f"Looking for pretrained model at {cfg.rl_pretrained_model_dir}", file=sys.stderr)
    loaded_model = model.load_model(cfg.rl_pretrained_model_dir)


    test_batch = np.vstack([[train_vocab.stoi[tk] for tk in stc.split(" ")] for stc in test_sentences]).astype(np.int)

    test_batch = torch.from_numpy(test_batch)

    if loaded_model == False:
        print("Failed to load checkpoints", file=sys.stderr)

    test_embeddings = model.embedding(test_batch)
    test_segments = model.critic(test_embeddings)

    segment_labels = torch.sigmoid(test_segments)
    
    for i in range(len(test_sentences)):
        print(test_sentences[i], segment_labels[i] > cfg.rl_critic_score_threshhold)
