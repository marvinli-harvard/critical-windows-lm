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
python experiments/chain_of_thought/run_revise_qa_critical_window.py --model_id meta-llama/Llama-3.1-8B-Instruct \
    --task math --answer_type math --cw_jump_threshold 0.5 --cw_decline_threshold -0.3 \
    --dataset_name competition_math --dataset_dir results/QANoiseDenoise/QANoiseDenoise_model=meta-llama-Llama-3.1-8B-Instruct_dataset=competition_math_split=test_nsamples=5000_num_per_noise=10
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
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=MAX_GEN_LEN_COT,help='Maximum number of tokens generated.')
    parser.add_argument('--max_answer_length', action="store", type=int, required=False, default=MAX_GEN_LEN_ANSWER,help='Maximum length of string to generate answer')
    parser.add_argument('--task', action="store", type=str, required=True, help='Type of questions.')
    parser.add_argument('--answer_type', action="store", type=str, required=True, 
                        choices=[e.value for e in AnswerType], help='Type of answer to generate')
    
    parser.add_argument('--cw_jump_threshold', action="store", type=float, required=True, help='Critical window parameter: threshold for jump.')
    parser.add_argument('--cw_decline_threshold', action="store", type=float, required=True,help='Critical window parameter: threshold for declining.')
    
    ## Dataset arguments
    parser.add_argument('--dataset_name', action="store", type=str, required=True, help='Name of dataset.')
    parser.add_argument('--dataset_dir', action="store", type=str, required=True, help='Location of dataset.')
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"ReviseCW_model={args.model_id.replace('/','-')}_dataset_name={args.dataset_name}_jump={args.cw_jump_threshold:.2f}_decline={args.cw_decline_threshold:.2f}"
        
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

    ## Load dataset, tokenizer, and model
    combined_df = pd.read_csv(os.path.join(args.dataset_dir, "dataset_with_gens_noisedenoise_ans.csv"))
    combined_df = combined_df[~combined_df.formatted_answer.isna()]

    uniq_stop_fracs = [0]+list(map(lambda x:float(x), list(combined_df.stop_frac.unique())))

    
    genwrapper = load_all(model_id=args.model_id, max_gen_length=args.max_gen_length, num_per_noise=1,seed=args.seed,max_answer_length=args.max_answer_length)
    ## Create prompt generator
    if args.model_id in ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.1-70B-Instruct","meta-llama/Llama-3.1-405B-Instruct"]:
        prompt_gen = LLAMAPromptGeneration(cot_prompt="", system_prompt="",tokenizer=genwrapper.tokenizer,
                                       clarify_choice_str=CLARIFY_CHOICE_STR_MATH if args.answer_type == "math" else CLARIFY_CHOICE_STR_MC)
    else:
        assert False, "Other types of model_ids are not supported"
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    qa_generator = QAGenerateNoiseDenoise(dataset=None,
                                          answer_type=args.answer_type, 
                                          genwrapper=genwrapper, 
                                          tokenizer=genwrapper.tokenizer,
                                          prompt_gen=prompt_gen)
    ############################################################################################################################################################################################################################################
    ## Build dataset with prompts
    ############################################################################################################################################################################################################################################
    combined_df["is_cw"] = False 
    percent_cons = combined_df.groupby(["problem","stop_frac"])[["is_consistent","is_right","is_stump"]].mean().reset_index()
    for problem in tqdm(percent_cons["problem"].unique()):
        problem_data = percent_cons[percent_cons["problem"] == problem]

        # Get stop_frac and is_consistent values
        is_consistent = problem_data["is_consistent"].values

        # Set alpha based on conditions
        if cw_condition(np.diff(is_consistent),cw=args.cw_jump_threshold,mon=args.cw_decline_threshold):
            combined_df.loc[combined_df.problem==problem,"is_cw"] = True
    
    combined_df["stop_frac_cw"]   = None
    combined_df["is_right_cw"]    = None
    for problem  in tqdm(combined_df.problem.unique()):
        problem_rows = combined_df[combined_df['problem'] == problem].sort_values(by='stop_frac')
        if problem_rows.is_cw.any():
            sf_after  = (problem_rows.groupby("stop_frac")["is_consistent"].mean().diff()>0.5).idxmax()
            sf_before = max([stop_frac for stop_frac in uniq_stop_fracs if stop_frac < sf_after])
            combined_df.loc[combined_df['problem'] == problem,"stop_frac_cw"] = sf_before
            combined_df.loc[combined_df['problem'] == problem,"is_right_cw"] = problem_rows.groupby("stop_frac")["is_right"].mean().loc[sf_before]
        else:
            combined_df.loc[combined_df['problem'] == problem,"is_right_cw"] = problem_rows["orig_is_right"].iloc[0].astype(float)
    
    cw_df = combined_df[combined_df.is_cw]

    cw_stump_string_ans = {}
    for problem in tqdm(cw_df.problem.unique()):
        stop_frac_cw = cw_df.groupby(["problem"]).stop_frac_cw.mean().loc[problem]
        stump_ans    = cw_df.loc[(cw_df.problem==problem)&(cw_df.stop_frac==round(stop_frac_cw,2))].stump_string_ans.drop_duplicates().values[0]
        cw_stump_string_ans[problem] = stump_ans.split("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nGiven all of the above")[0]+"<|start_header_id|>user<|end_header_id|>You may have made a mistake here. Think a little bit harder.<|eot_id|>\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    problems_df = (
                    cw_df[["problem","formatted_answer"]]
                    .drop_duplicates()
                    .merge(pd.Series(cw_stump_string_ans)
                        .reset_index()
                        .rename(columns={"index":"problem",0:"reflect_context"})
                        )
    )

    # Save cw_df and problems_df to CSV files
    combined_df.to_csv(os.path.join(args.experiment_dir, "df.csv"), index=False)
    cw_df.to_csv(os.path.join(args.experiment_dir, "cw_df.csv"), index=False)
    problems_df.to_csv(os.path.join(args.experiment_dir, "problems_df.csv"), index=False)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    ############################################################################################################################################################################################################################################
    ## Run generations, get new answers, and grade
    ############################################################################################################################################################################################################################################
    outputs = genwrapper.model.generate(problems_df["reflect_context"], genwrapper.sampling_first)

    new_answers = {}
    for i, output in enumerate(outputs):
        new_answers[i] = {
            "problem" :problems_df["problem"].iloc[i],
            "formatted_answer" :problems_df["formatted_answer"].iloc[i],
            "revise_input_string" :problems_df["reflect_context"].iloc[i],
            "revise_output_tokens" : output.prompt_token_ids+list(output.outputs[0].token_ids),
            "revise_output_string" : output.prompt+output.outputs[0].text, 
        }

    new_answers_completed = [
            prompt_gen.complete_with_answer(new_answers[i]["revise_output_tokens"])
            for i in range(len(outputs))
    ]
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    outputs, responses = qa_generator.generate(new_answers_completed, genwrapper.sampling_answer)

    for i in range(len(new_answers)):
        new_answers[i].update(qa_generator.format_answer("revise", new_answers_completed[i], outputs[i][0],responses[i][0]))
        new_answers[i].update({
            "revise_is_right":qa_generator.compare_answers(
                new_answers[i].get("revise_ans_format"),
                new_answers[i].get("formatted_answer")
            )
        })
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    # Save new_answers to a .pt file
    torch.save(new_answers, os.path.join(args.experiment_dir, "new_answers.pt"))

    # Create a pandas dataframe
    new_answers_df = pd.DataFrame.from_dict(new_answers, orient='index')

    # Save the dataframe to CSV
    new_answers_df.to_csv(os.path.join(args.experiment_dir, "new_answers_df.csv"), index=False)
    orig_columns = ["problem","formatted_answer",
                    "revise_input_string","revise_output_string",
                    "revise_ans_format","revise_string_ans",
                    "revise_ans_string",
                    "revise_is_right"]
    df = create_dataframe(new_answers.values(), orig_columns)
    df.to_csv(os.path.join(args.experiment_dir, "dataset_with_cw_revised.csv"), index=False)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()

