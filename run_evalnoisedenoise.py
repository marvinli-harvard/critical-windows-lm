import os
from datetime import datetime
import pandas as pd 
import time
import json
import argparse
import torch
from datasets import load_dataset, Dataset
from typing import Dict
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from accelerate.utils import set_seed
from utils import *
from grader_utils import *
from LLAMANoiseDenoise import LLAMANoiseDenoise
import sys
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Example command line prompt

python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type multiple_choice \
    --experiment_dir results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=truthfulqa-truthful_qa_split=validation_nsamples=2000     
"""

def main():
    ####################################################################################################
    # SETUP
    ####################################################################################################
    parser = argparse.ArgumentParser()
    ## Model arguments
    parser.add_argument('--model_id', action="store", type=str, required=True, help="Name of model")

    ## Experiment config details
    parser.add_argument('--experiment_dir', action="store", type=str, required=True, help='experiment dir.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=100,help='Number of samples.')
    parser.add_argument('--percent_prompt', action="store", nargs='+', type=float, required=False, 
                        default=[0.1,0.3,0.5,0.7,0.9], help='List of percent prompts to include')
    parser.add_argument('--answer_type', action="store", type=str, required=True, 
                        choices=['multiple_choice', 'math'], help='Type of answer to generate')
    parser.add_argument('--bs', action="store", type=int, required=False, default=1, help='Batch size.')
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=2243, help='Seed')
    args = parser.parse_args()
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    ## Save to JSON
    with open(f"{args.experiment_dir}/eval_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    start_time = time.time()

    ## Load model
    set_seed(args.seed)
    pipeline = load_model_pipeline(args.model_id)
    
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt="", system_prompt="",
                                       tokenizer=pipeline.tokenizer)
    else:
        assert False, "Other types of model_ids are not supported"
    print(f"Finished Loading pipeline")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Now, evaluate and save the answer 
    ############################################################################################################################################################################################################################################    
    data_path = os.path.join(args.experiment_dir, "response_with_noised_versions.pt")
    noised_data = torch.load(data_path,map_location=torch.device('cpu'))
    
    results = []
    for sample in tqdm(noised_data):
        noised_tokens = torch.tensor([prompt_gen.complete_with_answer(sample["full_response_tokens"].tolist(),
                                                                      pipeline.tokenizer)])
        original_tokens = pipeline.model.generate(noised_tokens.to(device), 
                                                  max_new_tokens=args.max_gen_length)
        running_dict = {
            "full_response_tokens_w_answer": original_tokens.cpu(),
            "full_response_string_w_answer": pipeline.tokenizer.decode(original_tokens[0])
        }

        new_tokens = original_tokens[0, noised_tokens.shape[-1]:]
        running_dict["new_tokens"] = new_tokens.cpu()
        running_dict["new_string"] = pipeline.tokenizer.decode(new_tokens)
        running_dict["answer"] = extract_answer(running_dict["new_string"],args.answer_type)

        for stop_frac in args.percent_prompt:
            running_dict[stop_frac] = {}
            noised_tokens = torch.tensor([prompt_gen.complete_with_answer(sample[stop_frac]["full_response_tokens"].tolist(),
                                                                          pipeline.tokenizer)])
            
            original_tokens = pipeline.model.generate(noised_tokens.to(device), 
                                                      max_new_tokens=args.max_gen_length)

            running_dict[stop_frac]["full_response_tokens_w_answer"] = original_tokens.cpu(),
            running_dict[stop_frac]["full_response_string_w_answer"] = pipeline.tokenizer.decode(original_tokens[0])
            
            new_tokens = original_tokens[0, noised_tokens.shape[-1]:]
            running_dict[stop_frac]["new_tokens"] = new_tokens.cpu()
            running_dict[stop_frac]["new_string"] = pipeline.tokenizer.decode(new_tokens)
            running_dict[stop_frac]["answer"] = extract_answer(running_dict[stop_frac]["new_string"],args.answer_type)
            running_dict[stop_frac]["is_same"] = compare_answers(running_dict[stop_frac]["answer"], running_dict["answer"], args.answer_type)
            
        results.append(running_dict.copy())
    print("Loading pipeline")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    

    # Save results
    results_path = os.path.join(args.experiment_dir, "evaluated_noisedenoise.json")
    torch.save(results,results_path)
    results = torch.load(os.path.join(args.experiment_dir, "evaluated_noisedenoise.json"),map_location=torch.device('cpu'))
    
    df = create_dataframe(results, args.percent_prompt)
    df.to_csv(os.path.join(args.experiment_dir, "noisedenoise_results.csv"), index=False)
    
    ## Create plots 
    generate_plots(df, args)

if __name__ == "__main__":
    main()

