#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=4-0:00:00
#SBATCH --mem=14000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH -o /home/rothenda/BMT/slurm_out/train_debug.%A.%a.%N.out
#SBATCH -J bmt-cap

python main.py > /home/rothenda/BMT/slurm_out/print.out --procedure train_rl_cap --rl_pretrained_model_dir /home/rothenda/BMT/log/train_rl_cap/1020140101/checkpoints/E_3 --mode BMHRL --rl_ff_a 256 --rl_ff_v 512 --rl_ff_c 1024 --rl_att_layers 3 --rl_gamma_manager 0.8 --rl_gamma_worker 0.7 --rl_warmstart_epochs 0 --rl_train_worker True --B 16 --video_features_path /nas/student/DanielRothenpieler/BMT/data/i3d_25fps_stack64step64_2stream_npy/ --audio_features_path /nas/student/DanielRothenpieler/BMT/data/vggish_npy/