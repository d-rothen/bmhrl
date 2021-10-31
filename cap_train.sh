#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=2-0:00:00
#SBATCH --mem=20000
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH -o /home/rothenda/BMT/slurm_out/train_debug.%A.%a.%N.out
#SBATCH -J bmt-cap

python -u main.py > /home/rothenda/BMT/slurm_out/print.out --procedure train_cap --B 32 --video_features_path /nas/student/DanielRothenpieler/BMT/data/i3d_25fps_stack64step64_2stream_npy/ --audio_features_path /nas/student/DanielRothenpieler/BMT/data/vggish_npy/
