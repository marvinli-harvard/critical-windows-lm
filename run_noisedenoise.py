import os
from datetime import datetime
import time
import json
import argparse
import itertools
import os
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset, Dataset
from typing import Dict
from torch.utils.data import DataLoader
from accelerate.utils import set_seed

from utils import *
from LLAMANoiseDenoise import LLAMANoiseDenoise
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Example command line prompts
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset competition_math --split train --bs 16 --task math --num_samples
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
    parser.add_argument('--cot_prompt', action="store", type=str, required=False, default = COT_PROMPT, help='Chain of thought prompt.')
    parser.add_argument('--system_prompt', action="store", type=str, required=False, default = SYSTEM_PROMPT, help='System prompt.')
    parser.add_argument('--percent_prompt', action="store", nargs='+', type=float, required=False, 
                        default=[0.1,0.3,0.5,0.7,0.9], help='List of percent prompts to include')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=2048,help='Number of samples.')
    parser.add_argument('--num_per_noise', action="store", type=int, required=False, default=1,help='Number of samples per noise level.')

    ## Dataset arguments
    parser.add_argument('--response_dataset', action="store", type=str, required=False, help='Location of dataset with responses.')
    parser.add_argument('--dataset', action="store", type=str, required=False, help='Name of dataset.')
    parser.add_argument('--split', action="store", type=str, required=True, help='Split of dataset.')
    
    parser.add_argument('--task', action="store", type=str, required=True, help='Type of questions.')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size.')

    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=2243, help='Seed')
    parser.add_argument('--model_half', action="store_true", required=False, help='Use half precision (fp16). 1 for use; 0 for not.')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    args.cot_prompt = args.cot_prompt.replace("/TASK/", args.task)
    args.system_prompt = args.system_prompt.replace("/TASK/", args.task)
    if args.experiment_name is None:
        if args.dataset:               
            args.experiment_name = f"NoiseDenoise_model={args.model_id.replace('/','-')}_dataset={args.dataset.replace('/','-').replace('@','-')}_split={args.split}_nsamples={args.num_samples}_num_per_noise={args.num_per_noise}"
        else:
            args.experiment_name = f"NoiseDenoise_model={args.model_id.replace('/','-')}_dataset=OLD_task={args.task}_split={args.split}_nsamples={args.num_samples}_num_per_noise={args.num_per_noise}"
        
    args.experiment_dir  = f"results/NoiseDenoise/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Loading dataset, model, and prompt generator
    ############################################################################################################################################################################################################################################
    print("Loading dataset, model, and prompt generator")
    set_seed(args.seed)

    ## Load dataset
    if args.response_dataset:
        first_responses   = torch.load(args.response_dataset)
    elif args.dataset:
        dataset = get_dataset(args=args)
    else:
        assert False, "Need to set either args.dataset, args.response_dataset, args.word"

    ## Load model
    pipeline = load_model_pipeline(args.model_id)
    if args.model_half:
        pipeline.half()
    pipeline.model.eval()

    ## Create prompt generator
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt=args.cot_prompt, 
                                       system_prompt=args.system_prompt,
                                       tokenizer=pipeline.tokenizer,
                                       clarify_choice_str="")
    else:
        assert False, "Other types of model_ids are not supported"
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Get first response to model if necesssary
    ############################################################################################################################################################################################################################################
    dataset = dataset.map(lambda example: {
                    **example,
                    "length" : len(prompt_gen.get_question_tokens(example["problem"]))
                    }
                ).sort("length")
    dataloader = DataLoader(dataset, batch_size = args.bs)    

    if args.response_dataset is None:
        print("Generating basic responses for model")
        first_responses = []
        for batch in tqdm(dataloader):
            # Convert all elements of the batch into valid question tokens
            question_tokens = [prompt_gen.get_question_tokens(item) for item in batch["problem"]]            
            inputs, outputs, responses = generate_tokens(pipeline,question_tokens, prompt_gen,max_gen_length=args.max_gen_length)
            for i in range(len(batch["problem"])):
                curr_value = {key: value[i] for key, value in batch.items()}
                curr_value["question_tokens"] = question_tokens[i]
                
                asst_index = torch.where(outputs[i]==prompt_gen.heading_to_tokens["assistant"])[0][0]+2
                curr_value["orig_tokens"] = outputs[i]
                curr_value["orig_string"] = responses[i]
                curr_value["asst_tokens"] = outputs[i][asst_index:]
                first_responses.append(curr_value.copy())

        # Save responses to a JSON file
        args.response_dataset = os.path.join(args.experiment_dir, "dataset_with_gens.pt")
        torch.save(first_responses,args.response_dataset)

        print(f"Saved to {args.response_dataset}")
        print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    ############################################################################################################################################################################################################################################
    ## Now, running the noising-denoising experiments
    ############################################################################################################################################################################################################################################    
    
    ## Form prefixes (noised results)
    noised_questions_tokens = []
    for gens in tqdm(first_responses):
        for copy, stop_frac in itertools.product(range(args.num_per_noise), args.percent_prompt):
            entry = gens.copy()
            entry["stop_frac"] = stop_frac
            entry["no"]        = copy
            inputs = prompt_gen.get_noise_denoise_question(question=entry["problem"], 
                                                        response_tokens=entry["asst_tokens"].tolist(),
                                                        stop_frac=stop_frac)
            entry["no_deno_input_tokens"] = list(inputs)
            entry["no_deno_input_string"] = pipeline.tokenizer.decode(inputs, skip_special_tokens=False)
            noised_questions_tokens.append(entry)
    
    ## Now run generation
    noised_questions_tokens = sorted(noised_questions_tokens, key=lambda x: len(x["no_deno_input_tokens"]))
    noised_denoised_results = []
    
    for batchno in tqdm(range(ceildiv(len(noised_questions_tokens), args.bs))):
        batch = noised_questions_tokens[(batchno*args.bs):min((batchno+1)*args.bs,len(noised_questions_tokens))]
        inputs, outputs, responses = generate_tokens(pipeline,
                                          [b["no_deno_input_tokens"] for b in batch], 
                                          prompt_gen,
                                          max_gen_length=args.max_gen_length)
        for i in range(len(batch)):
            curr_value = batch[i].copy()
            curr_value["no_deno_output_tokens"] = outputs[i]
            curr_value["no_deno_output_strings"] = responses[i]
            noised_denoised_results.append(curr_value)
    
    ## Convert the response_with_noised_versions to a dataset and save to disk
    noise_denoise_path = os.path.join(args.experiment_dir, "response_with_noised_versions.pt")
    torch.save(noised_denoised_results,noise_denoise_path)

    print(f"Final dataset with noised versions saved to {noise_denoise_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")




if __name__ == '__main__':
    main()

