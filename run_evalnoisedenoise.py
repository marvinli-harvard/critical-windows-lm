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

python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type math \
    --bs 4 --experiment_dir results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=competition_math_split=train_nsamples=8_num_per_noise=1/
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
    with open(os.path.join(args.experiment_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    start_time = time.time()

    ## Load model
    set_seed(args.seed)
    pipeline = load_model_pipeline(args.model_id)
    
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt="", system_prompt="",
                                       tokenizer=pipeline.tokenizer,
                                       clarify_choice_str= CLARIFY_CHOICE_STR_MATH if args.answer_type == "math" else CLARIFY_CHOICE_STR_MC)
    else:
        assert False, "Other types of models are not supported at the moment"
    print(f"Finished Loading pipeline")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################# 
    # Now, evaluate and save the answer 
    ###########################################################################
    data_path = os.path.join(args.experiment_dir, "response_with_noised_versions.pt")
    noised_data = torch.load(data_path,map_location=torch.device('cpu'))
    
    ## Run generation on original responses
    existing_orig_responses  = {}
    existing_stump_responses = {}
    for example in noised_data:
        if example["problem"] not in existing_orig_responses:
            existing_orig_responses[example["problem"]] = \
                prompt_gen.complete_with_answer(example["orig_tokens"].tolist())
        if tuple(example["no_deno_input_tokens"]) not in existing_stump_responses:
            existing_stump_responses[tuple(example["no_deno_input_tokens"])] = \
                prompt_gen.complete_with_answer(example["no_deno_input_tokens"] + [prompt_gen.heading_to_tokens["eot_id"]])
        
    ## Compute answers of original responses
    prompt_responses = sorted(existing_orig_responses.items(), key=lambda x: len(x[1]))    
    orig_answers     = {} 
    print("Computing answers of original responses")
    for batchno in tqdm(range(ceildiv(len(prompt_responses ),args.bs))):
        batch = prompt_responses[(batchno*args.bs):min((batchno+1)*args.bs,len(prompt_responses))]
        _, outputs, responses = generate_tokens(pipeline,[b[1]for b in batch], prompt_gen, max_gen_length=args.max_gen_length)
        for i in range(len(batch)):
            new_tokens = outputs[i][len(batch[i][1]):-1]
            new_string = pipeline.tokenizer.decode(new_tokens)
            orig_answers[batch[i][0]] = {
                "orig_tokens_ans": outputs[i],
                "orig_string_ans": responses[i],
                "orig_ans_tokens": new_tokens,
                "orig_ans_string": new_string,
                "orig_ans_format": extract_answer(new_string, args.answer_type),
            } 
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of stump versions")
    existing_stump_responses = sorted(existing_stump_responses.items(), key=lambda x: len(x[1]))    
    stump_answers     = {} 
    
    for batchno in tqdm(range(ceildiv(len(existing_stump_responses),args.bs))):
        batch = existing_stump_responses[(batchno*args.bs):min((batchno+1)*args.bs,len(existing_stump_responses))]
        _, outputs, responses = generate_tokens(pipeline,[b[1]for b in batch], prompt_gen, max_gen_length=args.max_gen_length)
        for i in range(len(batch)):
            new_tokens = outputs[i][len(batch[i][1]):-1]
            new_string = pipeline.tokenizer.decode(new_tokens)
            stump_answers[batch[i][0]] = {
                "stump_tokens_ans": outputs[i],
                "stump_string_ans": responses[i],
                "stump_ans_tokens": new_tokens,
                "stump_ans_string": new_string,
                "stump_ans_format": extract_answer(new_string, args.answer_type),
            } 


    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of noised denoised versions")   
    results = []
    noised_data = sorted(noised_data, 
                        key=lambda x: len(prompt_gen.complete_with_answer(x["no_deno_output_tokens"].tolist())))
    for batchno in tqdm(range(ceildiv(len(noised_data),args.bs))):
        batch = noised_data[(batchno*args.bs):min((batchno+1)*args.bs,len(noised_data))]
        noised_tokens = [
            prompt_gen.complete_with_answer(b["no_deno_output_tokens"].tolist()) 
            for b in batch
        ]
        _, outputs, responses = generate_tokens(pipeline,noised_tokens, prompt_gen, max_gen_length=args.max_gen_length)

        for i in range(len(batch)):
            curr_results = batch[i].copy()
            curr_results["no_deno_tokens_ans"] = outputs[i]
            curr_results["no_deno_string_ans"] = responses[i]

            new_tokens = outputs[i][len(noised_tokens[i]):-1]
            new_string = pipeline.tokenizer.decode(new_tokens)
            curr_results["no_deno_ans_tokens"]  = new_tokens
            curr_results["no_deno_ans_string"]  = new_string
            curr_results["no_deno_ans_format"]  = extract_answer(new_string, args.answer_type)
            curr_results.update(orig_answers[curr_results["problem"]].copy())
            curr_results.update(stump_answers[tuple(curr_results["no_deno_input_tokens"])].copy())
            
            curr_results["orig_is_right"]   = compare_answers(curr_results["orig_ans_format"],
                                                              curr_results["formatted_answer"],
                                                              args.answer_type)
            curr_results["stump_is_right"]   = compare_answers(curr_results["stump_ans_format"],
                                                              curr_results["formatted_answer"],
                                                              args.answer_type)
            curr_results["is_consistent"]   = compare_answers(curr_results["no_deno_ans_format"],
                                                              curr_results["orig_ans_format"],
                                                              args.answer_type)
            curr_results["is_right"]        = compare_answers(curr_results["no_deno_ans_format"],
                                                              curr_results["formatted_answer"],
                                                              args.answer_type)
            curr_results["is_stump"]        = compare_answers(curr_results["no_deno_ans_format"],
                                                              curr_results["stump_ans_format"],
                                                              args.answer_type)
            results.append(curr_results.copy())
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.experiment_dir, "evaluated_noisedenoise.pt")
    torch.save(results,results_path)

    orig_columns = ["problem","formatted_answer","stop_frac","no",
                'orig_ans_string',"orig_ans_format",
                'stump_ans_string',"stump_ans_format",
                "no_deno_ans_format",'no_deno_ans_string',
                "is_consistent","is_right","orig_is_right",
                "stump_is_right","is_stump"
                ]
    
    df = create_dataframe(results, orig_columns)
    df.to_csv(os.path.join(args.experiment_dir, "eval_noisedenoise.csv"), index=False)



if __name__ == "__main__":
    main()

