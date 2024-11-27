import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import argparse

import torch
from accelerate.utils import set_seed

from utils.utils import *
from utils.configuration import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *

"""
Runs generations on fixed dataset to produce noise denoise on a fixed dataset
python run_fixednoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset_type repeat_word --num_per_noise 10

python run_fixednoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset_type harmful_prefix --num_per_noise 10 --num_samples 10
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    ## Model arguments
    parser.add_argument('--model_id', action="store", type=str, required=True, help="Name of model")

    ## Experiment config details
    parser.add_argument('--experiment_name', action="store", type=str, required=False, help='Optional experiment name.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=REPEAT_WORD_MAX,help='Maximum generation length')
    parser.add_argument('--num_per_noise', action="store", type=int, required=False, default=1,help='Number of samples per `noise level`.')
    
    ## Dataset arguments
    parser.add_argument('--dataset_type', action="store", type=str, required=True, 
                        choices=[e.value for e in DatasetType], help='Type of fixed noise denoise dataset')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"FixedNoiseDenoise_model={args.model_id.replace('/','-')}_dataset_type={args.dataset_type}_num_samples={args.num_samples}_num_per_noise={args.num_per_noise}"        
    args.experiment_dir  = f"results/FixedNoiseDenoise/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Load model and dataset
    ############################################################################################################################################################################################################################################
    set_seed(args.seed)

    ## Construct dataset and load model
    model = load_all(args)
    fixed_wrapper = FixedNoiseDenoiseWrapper(dataset=args.dataset_type, args=args, tokenizer=model.tokenizer)
    dataset = fixed_wrapper.return_dataset()

    ############################################################################################################################################################################################################################################
    ## Run evaluation and grade each answer
    ############################################################################################################################################################################################################################################
    prompt_ids = [model.tokenizer.encode(dict_entry["context"][0]) for dict_entry in dataset.iter(batch_size=1)]
    _, responses = generate_tokens(model, prompt_ids, None, model.sampling_repeat)
    
    final_answers = []
    for i, val in tqdm(list(enumerate(dataset.iter(batch_size=1)))):
        for j in range(len(responses[i])):
            curr_value = {k:v[0] for k,v in val.items()}
            curr_value["answer"] = responses[i][j]
            curr_value["no"] = j
            final_answers.append(curr_value)    
    
    ## Delete model
    del model.model
    torch.cuda.empty_cache()

    final_answers = fixed_wrapper.grade(model_answers=final_answers)
    # Save final answers to a .pt file
    torch.save(final_answers, f"{args.experiment_dir}/graded_answers.pt")

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
