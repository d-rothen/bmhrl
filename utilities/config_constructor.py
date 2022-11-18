import os
from time import localtime, strftime
import torch
from shutil import copytree, ignore_patterns

class Config(object):
    '''
    Note: don't change the methods of this class later in code.
    '''

    def __init__(self, args):
        '''
        Try not to create anything here: like new forders or something
        '''
        self.curr_time = strftime('%y%m%d%H%M%S', localtime())

        self.procedure = args.procedure
        # dataset
        self.train_meta_path = args.train_meta_path
        self.val_1_meta_path = args.val_1_meta_path
        self.val_2_meta_path = args.val_2_meta_path
        self.modality = args.modality
        self.video_feature_name = args.video_feature_name
        self.audio_feature_name = args.audio_feature_name
        self.video_features_path = args.video_features_path
        self.audio_features_path = args.audio_features_path
        # make them d_video and d_audio
        self.d_vid = args.d_vid
        self.d_aud = args.d_aud
        self.start_token = args.start_token
        self.end_token = args.end_token
        self.pad_token = args.pad_token
        self.max_len = args.max_len
        self.min_freq_caps = args.min_freq_caps
        self.mode = args.mode

        #rl agent

        self.rl_low_level_enc_d = args.rl_low_level_enc_d
        self.rl_high_level_enc_d = args.rl_high_level_enc_d

        self.rl_worker_lstm = args.rl_worker_lstm
        
        self.rl_manager_lstm = args.rl_manager_lstm
        
        self.rl_goal_d = args.rl_goal_d
        self.rl_attn_d = args.rl_attn_d
        
        self.rl_critic_path = args.rl_critic_path
        self.rl_critic_score_threshhold = args.rl_critic_score_threshhold

        self.word_emb_caps = args.word_emb_caps
        self.unfreeze_word_emb = args.unfreeze_word_emb

        self.rl_pretrained_model_dir = args.rl_pretrained_model_dir
        self.rl_train_worker = args.rl_train_worker
        self.rl_warmstart_epochs = args.rl_warmstart_epochs
        self.rl_projection_d = args.rl_projection_d

        self.rl_gamma_worker = args.rl_gamma_worker
        self.rl_gamma_manager = args.rl_gamma_manager
        self.rl_reward_weight_worker = args.rl_reward_weight_worker
        self.rl_reward_weight_manager = args.rl_reward_weight_manager

        self.rl_att_layers = args.rl_att_layers
        self.rl_att_heads = args.rl_att_heads

        self.rl_ff_c = args.rl_ff_c
        self.rl_ff_v = args.rl_ff_v
        self.rl_ff_a = args.rl_ff_a
        
        self.rl_value_function_lr = args.rl_value_function_lr
        self.rl_cap_warmstart_lr = args.rl_cap_warmstart_lr
        self.rl_cap_lr = args.rl_cap_lr
        self.rl_stabilize = args.rl_stabilize

        self.dout_p = args.dout_p

        self.use_linear_embedder = args.use_linear_embedder
        if args.use_linear_embedder:
            self.d_model_video = args.d_model_video
            self.d_model_audio = args.d_model_audio
        else:
            self.d_model_video = self.d_vid
            self.d_model_audio = self.d_aud

        self.d_model = args.d_model
        self.d_model_caps = args.d_model_caps

        # training
        self.device_ids = args.device_ids
        self.device = f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu'
        self.train_batch_size = args.B * len(self.device_ids)
        self.inference_batch_size = args.inf_B_coeff * self.train_batch_size
        self.scheduler = args.scheduler
        self.epoch_num = args.epoch_num
        self.one_by_one_starts_at = args.one_by_one_starts_at
        self.early_stop_after = args.early_stop_after
        # criterion
        self.smoothing = args.smoothing  # 0 == cross entropy
        self.grad_clip = args.grad_clip
        # optimizer
        self.optimizer = args.optimizer
        if self.optimizer == 'adam':
            self.beta1, self.beta2 = args.betas
            self.eps = args.eps
            self.weight_decay = args.weight_decay
        else:
            raise Exception(f'Undefined optimizer: "{self.optimizer}"')

        # evaluation
        self.reference_paths = args.reference_paths
        self.tIoUs = args.tIoUs
        self.max_prop_per_vid = args.max_prop_per_vid
        self.prop_pred_path = args.prop_pred_path
        self.avail_mp4_path = args.avail_mp4_path
        # logging
        self.to_log = args.to_log
        if args.to_log:
            self.log_dir = os.path.join(args.log_dir, args.procedure)
            self.checkpoint_dir = self.log_dir  # the same yes
            # exper_name = self.make_experiment_name()
            exper_name = self.curr_time[2:]
            self.log_path = os.path.join(self.log_dir, exper_name)
            self.model_checkpoint_path = os.path.join(self.checkpoint_dir, exper_name)
        else:
            self.log_dir = None
            self.log_path = None

