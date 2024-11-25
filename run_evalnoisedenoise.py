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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Example command line prompt

python run_evalnoisedenoise.py --model_id meta-llama/Llama-3.1-8B-Instruct --answer_type math \
    --experiment_dir results/NoiseDenoise/NoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=competition_math_split=train_nsamples=8_num_per_noise=1/
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
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=MAX_GEN_LEN_ANSWER,help='Number of samples.')
    parser.add_argument('--answer_type', action="store", type=str, required=True, 
                        choices=[e.value for e in AnswerType], help='Type of answer to generate')
    
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    ## Save to JSON
    with open(os.path.join(args.experiment_dir, "eval_config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    args.num_per_noise = 1
    
    start_time = time.time()

    ## Load model
    set_seed(args.seed)
    ## Load tokenizer and model
    model = load_all(args)
    
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMANoiseDenoise(cot_prompt="", system_prompt="",
                                       tokenizer=model.tokenizer,
                                       clarify_choice_str=CLARIFY_CHOICE_STR_MATH if args.answer_type == "math" else CLARIFY_CHOICE_STR_MC)
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
        
    print("Computing answers of original responses")
    orig_responses   = list(existing_orig_responses.items())
    orig_tokens  = [item[1] for item in orig_responses] 
    orig_answers     = {} 
    outputs_orig, response_orig = generate_tokens(model, orig_tokens, prompt_gen, model.sampling_first)
    for i in range(len(orig_responses)):
        new_tokens = outputs_orig[i][0][len(orig_tokens[i]):-1]
        new_string = model.tokenizer.decode(new_tokens)
        orig_answers[orig_responses[i][0]] = {
            "orig_tokens_ans": outputs_orig[i][0],
            "orig_string_ans": response_orig[i][0],
            "orig_ans_tokens": new_tokens,
            "orig_ans_string": new_string,
            "orig_ans_format": extract_answer(new_string, args.answer_type),
        }
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of stump versions")
    stump_responses = list(existing_stump_responses.items())    
    stump_tokens  = [item[1] for item in stump_responses]    
    stump_answers     = {} 
    outputs_stump, response_stump = generate_tokens(model, stump_tokens, prompt_gen, model.sampling_first)
    for i in range(len(stump_responses)):
        new_tokens = outputs_stump[i][0][len(stump_tokens[i]):-1]
        new_string = model.tokenizer.decode(new_tokens)
        stump_answers[stump_responses[i][0]] = {
            "stump_tokens_ans": outputs_stump[i][0],
            "stump_string_ans": response_stump[i][0],
            "stump_ans_tokens": new_tokens,
            "stump_ans_string": new_string,
            "stump_ans_format": extract_answer(new_string, args.answer_type),
        }

    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    print("Computing answer of noised denoised versions")   
    results = []
    noised_data_tokens = [prompt_gen.complete_with_answer(b["no_deno_output_tokens"].tolist()) 
                          for b in noised_data]
    outputs_noise, response_noise = generate_tokens(model, noised_data_tokens, prompt_gen, model.sampling_first)
    for i in tqdm(range(len(noised_data))):
        curr_results = noised_data[i].copy()
        curr_results["no_deno_tokens_ans"] = outputs_noise[i][0]
        curr_results["no_deno_string_ans"] = response_noise[i][0]
        new_tokens = outputs_noise[i][0][len(noised_data_tokens[i]):-1]
        new_string = model.tokenizer.decode(new_tokens)

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
        curr_results["stump_is_consistent"]   = compare_answers(curr_results["stump_ans_format"],
                                                            curr_results["orig_ans_format"],
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
                "stump_is_right","is_stump","stump_is_consistent",
                "stump_string_ans","no_deno_string_ans","orig_string_ans"
                ]
    
    df = create_dataframe(results, orig_columns)
    df.to_csv(os.path.join(args.experiment_dir, "eval_noisedenoise.csv"), index=False)



if __name__ == "__main__":
    main()

