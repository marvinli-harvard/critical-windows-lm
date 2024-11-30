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

from prompt_generation.LLAMAPromptGeneration import LLAMAPromptGeneration
from generation.QAGenerateNoiseDenoise import QAGenerateNoiseDenoise

"""
Example command line prompts
python experiments/chain_of_thought/run_qa_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset competition_math --split train --task math --num_samples 8 --answer_type math
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
                        default=DEFAULT_PERCENT_PROMPT, help='List of percent prompts to include')
    parser.add_argument('--num_samples', action="store", type=int, required=False, help='Number of samples.')
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=MAX_GEN_LEN_COT,help='Number of samples.')
    parser.add_argument('--num_per_noise', action="store", type=int, required=False, default=1,help='Number of samples per noise level.')
    parser.add_argument('--task', action="store", type=str, required=True, help='Type of questions.')
    parser.add_argument('--answer_type', action="store", type=str, required=True, 
                        choices=[e.value for e in AnswerType], help='Type of answer to generate')
    
    ## Dataset arguments
    parser.add_argument('--dataset', action="store", type=str, required=False, help='Name of dataset.')
    parser.add_argument('--split', action="store", type=str, required=True, help='Split of dataset.')    
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    args.cot_prompt = args.cot_prompt.replace("/TASK/", args.task)
    args.system_prompt = args.system_prompt.replace("/TASK/", args.task)
    if args.experiment_name is None:
        if args.dataset:               
            args.experiment_name = f"QANoiseDenoise_model={args.model_id.replace('/','-')}_dataset={args.dataset.replace('/','-').replace('@','-')}_split={args.split}_nsamples={args.num_samples}_num_per_noise={args.num_per_noise}"
        else:
            args.experiment_name = f"QANoiseDenoise_model={args.model_id.replace('/','-')}_dataset=OLD_task={args.task}_split={args.split}_nsamples={args.num_samples}_num_per_noise={args.num_per_noise}"
        
    args.experiment_dir  = f"results/QANoiseDenoise/{args.experiment_name}/"
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

    ## Load dataset, tokenizer, and modle
    dataset = get_qa_dataset(dataset = args.dataset, split=args.split, num_samples = args.num_samples) 
    genwrapper = load_all(model_id=args.model_id, max_gen_length=args.max_gen_length, num_per_noise=args.num_per_noise)
    
    ## Create prompt generator
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMAPromptGeneration(cot_prompt=args.cot_prompt, 
                                       system_prompt=args.system_prompt,
                                       tokenizer=genwrapper.tokenizer,
                                       clarify_choice_str="")
    else:
        assert False, "Other types of model_ids are not supported"
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Get first response to model if necesssary
    ############################################################################################################################################################################################################################################
    qa_generator = QAGenerateNoiseDenoise(dataset=dataset,
                                          answer_type=args.answer_type, 
                                          genwrapper=genwrapper, 
                                          tokenizer=genwrapper.tokenizer,
                                          prompt_gen=prompt_gen)


    print("Generating basic responses for model")
    first_responses = qa_generator.generate_basic()
    
    # Save responses to a JSON file
    args.orig_gens_loc = os.path.join(args.experiment_dir, "dataset_with_gens.pt")
    torch.save(first_responses,args.orig_gens_loc)

    print(f"Saved to {args.orig_gens_loc}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    ############################################################################################################################################################################################################################################
    ## Now, running the noising-denoising experiments
    ############################################################################################################################################################################################################################################    
    
    ## Form prefixes (noised results)
    noised_denoised_results = qa_generator.generate_noised_denoised(percent_prompt=args.percent_prompt)

    ## Convert the response_with_noised_versions to a dataset and save to disk
    noise_denoise_path = os.path.join(args.experiment_dir, "dataset_with_gens_noisedenoise.pt")
    torch.save(noised_denoised_results,noise_denoise_path)

    print(f"Final dataset with noised versions saved to {noise_denoise_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


    ############################################################################################################################################################################################################################################
    ## Finally, grade all responses
    ############################################################################################################################################################################################################################################    
    ## Run generation on original responses

    qa_generator.generate_orig_stump()
        
    print("Computing answers of original responses")
    qa_generator.generate_original_answer()

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of stump versions")
    qa_generator.generate_stump_answers()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of noised denoised versions")   
    results = qa_generator.generate_noised_denoised_answer()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    # Save results
    results_path = os.path.join(args.experiment_dir, "dataset_with_gens_noisedenoise_ans.pt")
    torch.save(results,results_path)
    
    orig_columns = ["problem","formatted_answer","stop_frac","no",
                'orig_ans_string',"orig_ans_format",
                'stump_ans_string',"stump_ans_format",
                "no_deno_ans_format",'no_deno_ans_string',
                "is_consistent","is_right","orig_is_right",
                "stump_is_right","is_stump","stump_is_consistent",
                "stump_string_ans","no_deno_string_ans","orig_string_ans"
                ]
    
    df = create_dataframe(results, orig_columns)
    df.to_csv(os.path.join(args.experiment_dir, "dataset_with_gens_noisedenoise_ans.csv"), index=False)


if __name__ == '__main__':
    main()

