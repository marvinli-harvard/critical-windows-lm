from prm800k.grading.grader import grade_answer
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from typing import Dict
import os
import matplotlib.pyplot as plt 
import re
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]
SYSTEM_PROMPT = "Produce a correct solution to the following /TASK/ question."
COT_PROMPT = \
    "Think of the /TASK/ question thoroughly step by step. \
    Please only respond with the answer after reasoning thoroughly. \n\n"
CLARIFY_CHOICE_STR = "Given all of the above, what’s the single, most likely answer? Your answer should have the format \"The answer is $ANSWER\", where $ANSWER is your answer. \n\n"

def load_model_pipeline(model_id, 
                        generator=None,
                        temperature=0.6):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=device,
        temperature=temperature,
        token=ACCESS_TOKEN,
        low_cpu_mem_usage=False,
        generator=generator
    )
    return pipeline

def extract_between(before, after,text):
    return text.split(before)[-1].split(after)[0]

def text_to_response(sample,
                     pipeline,
                     system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=4096
    )
    return outputs[0]

def question_to_response(sample, pipeline,
                         system_prompt=SYSTEM_PROMPT):
    return text_to_response(sample["problem"], pipeline, system_prompt)


def get_raw_tokens_from_response(text :str , tokenizer):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def extract_answer(input: str, type_answer: str):
    if type_answer == "multiple_choice":
        # List of patterns to match different formats of answers
        patterns = [
            re.compile(r'answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?', re.IGNORECASE),
            re.compile(r'answer:\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?', re.IGNORECASE),
            re.compile(r'the answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?', re.IGNORECASE),
            re.compile(r'correct answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?', re.IGNORECASE),
            re.compile(r'option\s*\(?([A-Za-z])\)?', re.IGNORECASE),  # Matches "option (F)" or "Option F"
            re.compile(r'the answer is\s*option\s*\(?([A-Za-z])\)?', re.IGNORECASE),  # Matches "The answer is option F"
            re.compile(r'answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'answer:\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'correct answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'answer\s*[=:]\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'answer\s*-\s*\(?([A-Za-z])\)?', re.IGNORECASE)
        ]

        # Attempt to find matches for each pattern
        for pattern in patterns:
            match = pattern.search(input)
            if match:
                return f"({match.group(1).upper()})"  # Normalize the format as (A), (B), etc.

        # If no patterns match, raise an error
        # raise ValueError(f"No valid multiple-choice answer found in input: {input}")
        print(f"No valid multiple-choice answer found in input: {input}")
        return ""
    
    elif type_answer == "math":
        # Match phrases like 'answer is 42', 'answer: 3.14', etc.
        patterns = [
            re.compile(r'answer is\s*\$?\\boxed{([\d\.\-\+\/\*\^\(\)]+)}\$?', re.IGNORECASE),
            re.compile(r'answer:\s*\$?\\boxed{([\d\.\-\+\/\*\^\(\)]+)}\$?', re.IGNORECASE),
            re.compile(r'the answer is\s*\$?\\boxed{([\d\.\-\+\/\*\^\(\)]+)}\$?', re.IGNORECASE),
            re.compile(r'correct answer is\s*\$?\\boxed{([\d\.\-\+\/\*\^\(\)]+)}\$?', re.IGNORECASE),
            re.compile(r'answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer:\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'the answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'correct answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer\s*[=:]\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer\s*-\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer is\s*\\boxed{[(]?([A-Za-z])[\)]?}', re.IGNORECASE),
            re.compile(r'answer:\s*\\boxed{[(]?([A-Za-z])[\)]?}', re.IGNORECASE),
            re.compile(r'the answer is\s*\\boxed{[(]?([A-Za-z])[\)]?}', re.IGNORECASE),
            re.compile(r'correct answer is\s*\\boxed{[(]?([A-Za-z])[\)]?}', re.IGNORECASE),
            re.compile(r'answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer:\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'Answer:\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'the answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'correct answer is\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer\s*[=:]\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE),
            re.compile(r'answer\s*-\s*([\d\.\-\+\/\*\^\(\)]+)', re.IGNORECASE)
        ]

        # Attempt to find matches for each pattern
        for pattern in patterns:
            match = pattern.search(input)
            if match:
                return match.group(1)  # Return the mathematical answer as is

        # If no patterns match, raise an error
        # raise ValueError(f"No valid math answer found in input: {input}")
        print(f"Unsupported answer type: {type_answer}")
        return ""
    
    else:
        # If an unsupported type_answer is provided, raise an error
        # raise ValueError(f"Unsupported answer type: {type_answer}")
        print(f"Unsupported answer type: {type_answer}")
        return ""
    
def compare_answers(answer1: str, answer2: str, type_answer: str) -> bool:
    if type_answer == "multiple_choice":
        return answer1.strip().upper() == answer2.strip().upper()
    elif type_answer == "math":
        return grade_answer(answer1, answer2)
    else:
        raise ValueError(f"Unsupported answer type: {type_answer}")
    
def create_dataframe(results, percent_prompts):
    """
    Creates a pandas DataFrame from the results with the following columns:
    - question: Original input question or prompt
    - original_response: Model's original response
    - new_response_{stop_frac}: Model's response with noised inputs for each stop_frac
    - is_same_{stop_frac}: Boolean indicating if the response matches the original response for each stop_frac
    """
    data = []
    for result in results:
        row = {
            "question": result.get("problem"),
            "original_response": result.get("full_response_string_w_answer"),
            "original_answer": result.get("answer"),
        }
        # Add new response and is_same for each stop_frac
        for stop_frac in percent_prompts:
            row[f"new_response_{stop_frac}"] = result[stop_frac].get("full_response_string_w_answer")
            row[f"is_same_{stop_frac}"] = result[stop_frac].get("is_same")
            row[f"answer_{stop_frac}"] = result[stop_frac].get("answer")
        
        data.append(row)
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    return df

def calculate_means_and_errors(df, percent_prompts, filter_inconsistent=False):
    """
    Calculate mean and standard error for 'is_same' columns using pandas.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        percent_prompts (list): List of prompt cutoff percentages.
        filter_inconsistent (bool): Whether to filter for inconsistent prompts.
        
    Returns:
        means (list): List of mean percentages for each cutoff.
        errors (list): List of standard errors for each cutoff.
    """
    if filter_inconsistent:
        # Filter for cases where 'is_same' is not true across all percent_prompts
        df = df[~df[[f'is_same_{stop_frac}' for stop_frac in percent_prompts]].all(axis=1)]
    
    means = []
    errors = []
    for stop_frac in percent_prompts:
        col_name = f'is_same_{stop_frac}'
        mean = df[col_name].mean() * 100  # Convert to percentage
        # Standard error using pandas
        error = df[col_name].std() / np.sqrt(len(df[col_name])) * 100
        means.append(mean)
        errors.append(error)
    
    return means, errors

def plot_and_save(means, errors, percent_prompts, title, ylabel, save_path):
    """
    Create and save a plot with error bars.
    
    Args:
        means (list): Mean percentages for each cutoff.
        errors (list): Standard errors for each cutoff.
        percent_prompts (list): List of prompt cutoff percentages.
        title (str): Title of the plot.
        ylabel (str): Y-axis label.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(percent_prompts, means, yerr=errors, fmt='-o', capsize=5, capthick=2, elinewidth=1)
    plt.xlabel("Percentage of Model Response Included")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(save_path, format="png", dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# Main logic for generating and saving plots
def generate_plots(df, args):
    """
    Generate both overall consistency and filtered consistency breakdown plots.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        args: Arguments object containing the experiment_dir and percent_prompt.
    """
    # Overall consistency plot
    overall_means, overall_errors = calculate_means_and_errors(df, args.percent_prompt)
    overall_plot_path = os.path.join(args.experiment_dir, "consistency_by_prompt_cutoff.png")
    plot_and_save(
        overall_means,
        overall_errors,
        args.percent_prompt,
        "Consistency of Responses by Prompt Cutoff Level",
        "Percentage of same responses",
        overall_plot_path
    )
    
    # Filtered inconsistency breakdown plot
    filtered_means, filtered_errors = calculate_means_and_errors(df, args.percent_prompt, filter_inconsistent=True)
    filtered_plot_path = os.path.join(args.experiment_dir, "consistency_breakdown_not_always_right.png")
    plot_and_save(
        filtered_means,
        filtered_errors,
        args.percent_prompt,
        "Consistency Breakdown by Model Responses",
        "Percentage of same responses",
        filtered_plot_path
    )
