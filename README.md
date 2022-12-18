# BMHRL

The **B**i**M**odal **H**ierarchical **R**einforcement **L**earning Agent is a hierarchical captioning model using audiovisual data to generate captions for videos.
It has been trained on the [ActivityNetCaptions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset, achieving a Meteor Score of 10.80.


• [ArXiv](https://arxiv.org/)

![Model](model.png)

# Getting Started
----
## 1. Clone the repository


```bash
git clone --recursive https://gitlab.com/
```

## 2. Download features & checkpoints

Download features (I3D and VGGish) and word embeddings (GloVe). The script will download them (~10 GB) and unpack into `./data` and `./.vector_cache` folders. *Make sure to run it while being in BMHRL folder*

```bash
bash ./download_data.sh
```

Set up a `conda` environment
```bash
conda env create -f ./req.yml
conda activate bmhrl
python -m spacy download en
```
Dowload the critic checkpoint
https://drive.google.com/file/d/1fidxz-WocOTsXN0gWnWpHl3VWEiKz0NX/view?usp=sharing

*(optionally)*
Dowload a baseline to warmstart the model:
https://drive.google.com/drive/folders/13I6BW4SreXEQgmFLgyCok9K-qOnOyYkY?usp=sharing

Download the best performing model:
https://drive.google.com/drive/folders/1zULCCntv8ZdQ3-n-EhmlZLLdJJsHIyDP?usp=sharing

## 3. Run the model

### Important Parameters:
- procedure : [train_rl_cap]
- mode : [BMHRL] 
- rl_warmstart_epochs: ( # of  epochs the model trains with standard KL Divergence )
- rl_pretrained_model_dir: ( directory of the checkpoints the model is initialised with)
- rl_train_worker: [True, False] ( model trains the worker before training the manager )
- B: ( batchsize )
- video_features_path: ( path of downloaded video features )
- audio_features_path: ( path of downloaded audio features )
- rl_critic_path: ( path of the downloaded critic checkpoint file )
- rl_pretrained_model_dir ( path of a saved model checkpoint folder )

Example:
```bash
python main.py --procedure train_rl_cap --mode BMHRL --rl_warmstart_epochs 2 --rl_pretrained_model_dir /home/xxxx/BMHRL/log/train_rl_cap/baseline/checkpoints/E_3 --rl_train_worker True --B 16 --rl_critic_path /home/xxxx/BMHRL/data/critic.cp  --video_features_path /nas/BMHRL/data/i3d_25fps_stack64step64_2stream_npy/ --audio_features_path /nas/BMHRL/data/vggish_npy/
```