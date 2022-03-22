# Multimodal Reinforcement based Dense Video Captioning

- [Multimodal Reinforcement based Dense Video Captioning](#dense-video-captioning-with-bi-modal-transformer)
  - [Summary](#summary)
  - [Structure](#structure)
  - [Getting Started](#getting-started)
  - [Train](#train)
  - [Viewing Results](#viewing-results)
  
## Summary
This project aims to utilize both audio and visual modalities for video captioning in a hierarchical reinforcement learning setting.
The HRL approach aims to better introduce sentence dynamics into the predicition process, such as sentence structure and focusing on vital information in sequences by learning so called "goals"
### Structure
The model consists of a stochastical policy worker agent that produces a probability distribution over the vocabulary given a generated goal, a representation of the multimodal features and the previous predicted word.

The goal is given by a manager network that too operates on a representation of the given multimodal features.

In order for the manager network to discern wether a goal has been reached by the worker agent, a critic network (pretrained on the CHARADE caption dataset) labels the given prediction sequence with 0 or 1 for each word, depending on a word being the final word to finish a certain goal.

Upon reaching a goal, the manager will generate a new goal for the worker to operate on.

The whole network is warmstarted with cross entropy loss on GT-sentences in order to enable the worker aswell as the manager to start off with a reasonable approach towards goal generation and word prediciton.


## Getting Started
Clone the repository. Mind the `--recursive` flag to make sure `submodules` are also cloned (evaluation scripts for Python 3 and scripts for feature extraction).
```bash
git clone --recursive https://git.rz.uni-augsburg.de/amiripsh/projektmodul-daniel-rothenpieler.git
```

Download features (I3D and VGGish) and word embeddings (GloVe). The script will download them (~10 GB) and unpack into `./data` and `./.vector_cache` folders. *Make sure to run it while being in BMT folder*
```bash
    bash ./download_data.sh
```

Set up a `conda` environment
```bash
conda env create -f ./conda_env.yml
conda activate bmt
# install spacy language model. Make sure you activated the conda environment
python -m spacy download en
```

## Train

The training is seperated in several stages.
First, the agent is warmstarted by applying cross-entropy loss to the predicted probability of ground truth sentences.
Secondly either the modules **Worker** or **Manager** will be trained in alternating phases.
This will reduce instability due to the ambilateral dependency of the Worker and Manager.

- *Train the captioning module*. Make sure to adjust the paths given in the script above to your own local paths.
```bash
sbatch cap_train.sh
```

## Viewing Results
Results will be saved in the ``./log/train_rl_cap/`` directory.
The latest best performing result is available in this repository at ``./results/logs/``.
As such they can be viewed via
```bash
tensorboard --logdir ./results/logs/
```