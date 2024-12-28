import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import torch
from accelerate.utils import set_seed

from utils.utils import *
from utils.configuration import *
from utils.dataset_utils import *
from utils.generation_utils import *
from utils.grader_utils import *

from generation.GenerateEvalBase import *
from generation.GenerateEvalExamples import *

"""
Runs generations to produce critical windows on structured data
python experiments/structured_output/run_structured_output_experiments.py --model_id meta-llama/Llama-3.1-8B-Instruct --num_samples 10000
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
    parser.add_argument('--max_gen_length', action="store", type=int, required=False, default=MAX_GEN_LEN_COT,help='Maximum generation length')
    parser.add_argument('--story_str', action="store", type=str, required=False, default=DEFAULT_STRUCTURED_STORY_PROMPT,help='structured story')
    parser.add_argument('--num_samples', action="store", type=int, required=False, default=10000, help='Number of samples.')
    parser.add_argument('--step_size', action="store", type=float, required=False, default=0.01, help='Step size for plotting critical windows.')
    
    # Device Arguments
    parser.add_argument('--seed', action="store", type=int, required=False, default=DEFAULT_SEED, help='Seed')
    args = parser.parse_args()
    
    start_time = time.time()

    ## Construct config file and save 
    if args.experiment_name is None:
        args.experiment_name = f"StructuredNoiseDenoise_model={args.model_id.replace('/','-')}_num_samples={args.num_samples}"        
    args.experiment_dir  = f"results/StructuredNoiseDenoise/{args.experiment_name}/"
    args.date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    os.makedirs(os.path.dirname(args.experiment_dir), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(args.experiment_dir),"ex_hierarchy"), exist_ok=True)
    
    # Save to JSON
    with open(f"{args.experiment_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    
    ############################################################################################################################################################################################################################################
    ## Load model and dataset
    ############################################################################################################################################################################################################################################
    set_seed(args.seed)

    ## Construct dataset and load model
    genwrapper = load_all(model_id=args.model_id,
                          max_gen_length=args.max_gen_length,
                          num_per_noise=args.num_samples,
                          seed=args.seed)
    dataset = create_structured_form_dataset(tokenizer=genwrapper.tokenizer,
                                            template_str=args.story_str)

    generator = GenerateEvalBase(genwrapper=genwrapper, 
                                 tokenizer=genwrapper.tokenizer)
    generated_responses = generator.generate_responses(dataset=dataset, 
                                                       sampling_params=genwrapper.sampling_repeat)
    
    # Save generations to a .pt file
    torch.save(generated_responses, f"{args.experiment_dir}/structured_responses.pt")

    print(f"Experiment completed in {time.time() - start_time:.2f} seconds.")

    ############################################################################################################################################################################################################################################
    ## Do some plotting
    ############################################################################################################################################################################################################################################
    structured_df = pd.DataFrame(generated_responses)
    structured_df["asst_response"]   = structured_df.response_string.apply(lambda x:extract_first_assistant_response(x))
    structured_df["choices"]         = structured_df.asst_response.apply(lambda x:tuple(extract_structured_choices(args.story_str,x)))
    structured_df["choices_tuple"]   = structured_df.choices.apply(lambda x:tuple([y[0] for y in x]))
    
    choices_df = structured_df[["asst_response","choices","choices_tuple"]].drop_duplicates()
    
    ## For every string, compute critical window hierarchy for them
    percent_values = list(np.arange(0,1,args.step_size))

    for i, final_str in tqdm(list(enumerate(choices_df.asst_response.unique()))):
        plot_critical_windows_given_str(percent_values, final_str, args.story_str,
                                        structured_df, args.step_size, 
                                        f"{args.experiment_dir}/ex_hierarchy/ex.{i}.png")

    structured_df.to_csv(f"{args.experiment_dir}/structured_responses.csv", index=False)
    (structured_df
     .choices_tuple
     .value_counts()
     .reset_index()
     .to_csv(f"{args.experiment_dir}/structured_value_counts.csv", 
             index=False))
    choices_df.to_csv(f"{args.experiment_dir}/structured_choices.csv", index=False)
    
    print(f"Plotting and saving completed in {time.time() - start_time:.2f} seconds.")

def plot_critical_windows_given_str(percent_values : list[float],
                                    final_str : str,
                                    story_prompt : str,
                                    structured_df : pd.DataFrame,
                                    step_size : float,
                                    save_name : float
                                    ):
    # Example data for percent-to-same curve
    percent_to_same = percent_values
    percent_to_same_values = pd.Series({per:percent_to_freq(per,structured_df,final_str) for per in percent_values})

    # Define the critical points for Tlower and Tupper
    tlower_points, tupper_points = find_normalized_positions(story_prompt, final_str, step_size)
    uniq_tuple = structured_df[structured_df.asst_response==final_str].choices_tuple.unique()[0]

    # Corresponding labels
    labels = [
        ",".join(list(uniq_tuple)[:l])
        for l in range(1,len(list(uniq_tuple))+1)
    ]

    # Plot the main data (percent-to-same curve)
    plt.figure(dpi=200)
    plt.plot(percent_to_same, percent_to_same_values, color="black")
    
    # Add Tlower and Tupper as dots on the existing graph
    plt.scatter(tlower_points, match_rows_with_precision(percent_to_same_values, tlower_points, precision=1e-4), 
                color='orange', label="Tbefore", marker='o', zorder=5)  # Slightly above
    plt.scatter(tupper_points,  match_rows_with_precision(percent_to_same_values, tupper_points, precision=1e-4), 
                color='blue', label="Tafter", marker='x', zorder=5)  # Slightly above

    # Add labels between the pairs
    for i, label in enumerate(labels):
        if i < len(labels)-1:
            # Between orange and blue points
            x_start = tupper_points[i]
            x_end = tlower_points[i+1]
            y_loc = percent_to_same_values.loc[tupper_points[i]]
        else:
            x_start = tupper_points[i]+0.05
            x_end = tupper_points[i]+0.05
            y_loc = 0.9
        # Add the label in the middle of the interval
        plt.text(
            (x_start + x_end) / 2,
            y_loc,  # Adjust the y-position for readability
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
        )
    # Add labels and legend
    plt.title("Critical windows for structured data generation")
    plt.xlabel("Frac of generation remaining")
    plt.ylabel("Frac consistent with original answer")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    main()
