#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH -C gmem12
#SBATCH --job-name=test_scales
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
module list                            # Have Nvidia tell us the GPU/CPU mapping so we know
module load cuda/11.7
module load decord-cuda-11-7
nvidia-smi topo -m
#source activate ABM

#python3 main.py --gpu 0 --run_id MEVID_resnet --run_description "train resnet model on the MEVID dataset" --dataset MEVID --model_version 'resnet' --input_dim 224 --batch_size 32 --num_epochs 500 --num_workers 4 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

#python3 main.py --gpu 0 --run_id MEVID_v1 --run_description "train V1 model on the MEVID dataset" --dataset MEVID --model_version 'v1' --input_dim 224 --num_heads 8 --num_layers 3 --batch_size 64 --num_epochs 500 --num_workers 4 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM --hidden_dim 2048 

python3 misc.py
