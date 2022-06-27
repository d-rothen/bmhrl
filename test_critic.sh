#!/bin/sh

#SBATCH --partition=dfl
#SBATCH --time=4-0:00:00
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --get-user-env
#SBATCH --export=ALL
#SBATCH -o /home/rothenda/BMT/slurm_out/test-critic/test.%A.%a.%N.out
#SBATCH -J test-critic

python main.py --procedure test_critic --rl_pretrained_model_dir /home/rothenda/BMT/results/logs/0513/0512140216/checkpoints/E_37 --rl_train_worker False --B 8 --video_features_path /nas/student/DanielRothenpieler/BMT/data/i3d_25fps_stack64step64_2stream_npy/ --audio_features_path /nas/student/DanielRothenpieler/BMT/data/vggish_npy/
