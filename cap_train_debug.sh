#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=2-0:00:00
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --exclude=eihw-gpu[02-03]
#SBATCH --nodelist=eihw-gpu1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH -o train_debug.out
#SBATCH -J mdvc-train

python -m debugpy --wait-for-client --listen 5999 main.py --procedure train_cap --B 32  --video_features_path /nas/student/DanielRothenpieler/BMT/data/i3d_25fps_stack64step64_2stream_npy/ --audio_features_path /nas/student/DanielRothenpieler/BMT/data/vggish_npy/
