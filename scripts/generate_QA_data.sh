#!/bin/bash
#SBATCH --job-name=generate_QA_data
#SBATCH -t 0-24:00                 # Runtime in D-HH:MM
#SBATCH --mem=200000               # Memory pool for all cores
#SBATCH -p gpu,gpu_requeue,seas_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-6
#SBATCH --mail-user=marvinli@college.harvard.edu
#SBATCH -o /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/output/myoutput_%A_%a.out   # STDOUT with task-specific path
#SBATCH -e /n/netscratch/sitanc_lab/Lab/mfli/slurm_logs/logs/err/myerrors_%A_%a.err    # STDERR with task-specific path

# Run the appropriate command based on the task ID
task_id=${SLURM_ARRAY_TASK_ID}

# Run the appropriate command based on the task ID
if [ "$task_id" -eq 0 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset competition_math --split train --task math --num_samples 10000 --answer_type math
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset competition_math --split train --task math --num_samples 400 --num_per_noise 25 --answer_type math
elif [ "$task_id" -eq 1 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset lucasmccabe/logiqa --split train --task logic --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset lucasmccabe/logiqa --split train --task logic --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
elif [ "$task_id" -eq 2 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset truthfulqa/truthful_qa --split validation  --task "true or false" --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset truthfulqa/truthful_qa --split validation  --task "true or false" --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
elif [ "$task_id" -eq 3 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split auxiliary_train --task "multiple choice" --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split auxiliary_train --task "multiple choice" --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
elif [ "$task_id" -eq 4 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split train  --task science --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split train  --task science --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
elif [ "$task_id" -eq 5 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Easy --split train --task science --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Easy --split train --task science --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
elif [ "$task_id" -eq 6 ]; then
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset deepmind/aqua_rat --split train  --task math --num_samples 10000 --answer_type multiple_choice
    python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset deepmind/aqua_rat --split train  --task math --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $task_id"
fi


