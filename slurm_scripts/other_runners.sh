#!/bin/bash
#SBATCH -t 0-24:00                 # Runtime in D-HH:MM
#SBATCH --mem=100000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split auxiliary_train --bs 1 --task "multiple choice" --num_samples 2000
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split train --bs 1 --task science --num_samples 1999

