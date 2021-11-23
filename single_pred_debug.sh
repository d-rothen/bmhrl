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

python -m debugpy --wait-for-client --listen 5999 ./sample/single_video_prediction.py \
    --prop_generator_model_path /home/rothenda/BMT/log/train_prop/baseline/best_prop_model.pt \
    --pretrained_cap_model_path /home/rothenda/BMT/log/train_cap/baseline/best_cap_model.pt \
    --vggish_features_path ./sample/women_long_jump_vggish.npy \
    --rgb_features_path ./sample/women_long_jump_rgb.npy \
    --flow_features_path ./sample/women_long_jump_flow.npy \
    --duration_in_secs 35.155 \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4