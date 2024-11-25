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

from NoiseDenoise.LLAMANoiseDenoise import *

"""
Example command line prompts
python run_noisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --dataset competition_math --split train --task math --num_samples 8
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
    
    ## Dataset arguments
    parser.add_argument('--response_dataset', action="store", type=str, required=False, help='Location of dataset with responses.')
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

    ## Load tokenizer and model
    model = load_all(args)
    ## Create prompt generator
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt=args.cot_prompt, 
                                       system_prompt=args.system_prompt,
                                       tokenizer=model.tokenizer,
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

    if args.response_dataset is None:
        print("Generating basic responses for model")
        first_responses = []
        question_tokens = [prompt_gen.get_question_tokens(item) for item in dataset["problem"]]
        outputs, responses = generate_tokens(model, question_tokens, prompt_gen,model.sampling_first)
        
        for i, val in enumerate(dataset.iter(batch_size=1)):
            curr_value = {k:v[0] for k,v in val.items()}
            asst_index = torch.where(outputs[i][0]==prompt_gen.heading_to_tokens["assistant"])[0][0]+2
            
            curr_value["question_tokens"] = question_tokens[i]
            curr_value["orig_tokens"] = outputs[i][0]
            curr_value["orig_string"] = responses[i][0]
            curr_value["asst_tokens"] = outputs[i][0][asst_index:]
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
        for stop_frac in args.percent_prompt:
            entry = gens.copy()
            entry["stop_frac"] = stop_frac
            inputs = prompt_gen.get_noise_denoise_question(question=entry["problem"], 
                                                           response_tokens=entry["asst_tokens"].tolist(),
                                                           stop_frac=stop_frac)
            entry["no_deno_input_tokens"] = list(inputs)
            entry["no_deno_input_string"] = model.tokenizer.decode(inputs, skip_special_tokens=False)
            noised_questions_tokens.append(entry)
    
    ## Now run generation
    question_tokens = [item["no_deno_input_tokens"] for item in noised_questions_tokens]
    outputs, responses = generate_tokens(model, question_tokens, prompt_gen, model.sampling_end)
    noised_denoised_results = []
    for i in tqdm(range(len(outputs))):    
        batch_output   = outputs[i]
        batch_response = responses[i]
        for j in range(len(batch_output)):
            curr_value = noised_questions_tokens[i].copy()
            curr_value["no_deno_output_tokens"]     = batch_output[j]
            curr_value["no_deno_output_strings"]    = batch_response[j]
            curr_value["no"]                        = j
            noised_denoised_results.append(curr_value)
    
    ## Convert the response_with_noised_versions to a dataset and save to disk
    noise_denoise_path = os.path.join(args.experiment_dir, "response_with_noised_versions.pt")
    torch.save(noised_denoised_results,noise_denoise_path)

    print(f"Final dataset with noised versions saved to {noise_denoise_path}")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")




if __name__ == '__main__':
    main()

