## MATH
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset competition_math --split test --task math --num_samples 10000 --answer_type math
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset competition_math --split test --task math --num_samples 400 --num_per_noise 25 --answer_type math

## LogiQA
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset lucasmccabe/logiqa --split test --task logic --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset lucasmccabe/logiqa --split test --task logic --num_samples 400 --num_per_noise 25 --answer_type multiple_choice

## TruthfulQA
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset truthfulqa/truthful_qa --split validation  --task "true or false" --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset truthfulqa/truthful_qa --split validation  --task "true or false" --num_samples 400 --num_per_noise 25 --answer_type multiple_choice

## MMLU
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split test --task "multiple choice" --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset cais/mmlu --split test --task "multiple choice" --num_samples 400 --num_per_noise 25 --answer_type multiple_choice

## ARC-Challenge
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split test  --task science --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Challenge --split test  --task science --num_samples 400 --num_per_noise 25 --answer_type multiple_choice

## ARC-Easy
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Easy --split test --task science --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset allenai/ai2_arc@ARC-Easy --split test --task science --num_samples 400 --num_per_noise 25 --answer_type multiple_choice

## AQUA-RAT
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset deepmind/aqua_rat --split test  --task math --num_samples 10000 --answer_type multiple_choice
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --dataset deepmind/aqua_rat --split test  --task math --num_samples 400 --num_per_noise 25 --answer_type multiple_choice
