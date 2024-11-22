from prm800k.grading.grader import grade_answer
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, load_dataset
import re
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

CLARIFY_CHOICE_STR_MC = "Given all of the above, what’s the single, most likely answer? Your answer should have the format \"The answer is ANSWER\", where ANSWER is your answer. \n\n"
CLARIFY_CHOICE_STR_MATH = "Given all of the above, what’s the single, most likely answer? Simplify it completely. Your answer should have the format \"The answer is $ANSWER$\", where ANSWER is your answer in LaTeX. \n\n"
REPEAT_WORD_LIST = [
    "company", "one", "b", "j", "life", "send", "make", "part", "with", "work", "word", "cell", "you",
    "time", "eye", "of", "on", "come", "good", "do", "up", "last", "year", "call", "a", "d", "out",
    "x", "the", "world", "new", "book", "day", "have", "their", "take", "in", "was", "different", 
    "point", "great", "man", "some", "person", "y", "v", "case", "-", "w", "\\", "my", "^", "i", 
    "*", "see", "first", "say", "he", "poem", "p", "would", "fact", "m", "as", "(", "c", "are", 
    "about", "early", "place", "q", "right", "g", "number", "think", "#", "hand", "problem", "f", 
    "$", "be", "for", "e", "it", "go", "k", "long", "!", "z", "is", "way", "and", ")", "|", "get", 
    "thing", "r", "n", "&", "that", "know", "t", "o", "to", "µ", "h"
]

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
    pipeline.tokenizer.padding_side = "left"
    pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
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
            "original_answer": result.get("formatted_answer"),
        }
        # Add new response and is_same for each stop_frac
        for stop_frac in percent_prompts:
            row[f"new_response_{stop_frac}"] = result[stop_frac].get("full_response_string_w_answer")
            row[f"is_same_{stop_frac}"] = result[stop_frac].get("is_same")
            row[f"answer_{stop_frac}"] = result[stop_frac].get("formatted_answer")
        
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

def num_to_chr(x):
    return f"({chr(65+x)})"
add_parans = lambda x : f"({x})"

def get_dataset(args):
    """
    Load and preprocess a dataset based on the provided arguments.
    Args:
        args (Namespace): A namespace object containing the following attributes:
            - dataset (str): The name of the dataset to load.
            - split (str): The dataset split to load (e.g., 'train', 'test').
            - seed (int, optional): A seed for shuffling the dataset (used for 'cais/mmlu').
            - num_samples (int, optional): The number of samples to select from the dataset (used for 'cais/mmlu').
    Returns:
        Dataset: A Hugging Face dataset object with the specified preprocessing applied.
    Raises:
        AssertionError: If the specified dataset is not supported.
    """

    if args.dataset == "lucasmccabe/logiqa":
        dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["context"] + " " + example["query"] + "\n " \
                        + "\n ".join([f"{num_to_chr(i)} {opt}" for i, opt in enumerate(example["options"])]),
                    "formatted_answer": num_to_chr(example["correct_option"]),
                    },
                )
    elif args.dataset == "truthfulqa/truthful_qa":    
        dataset = load_dataset(args.dataset,"multiple_choice",split=args.split,trust_remote_code=True)
        def scramble_list(x, return_randperm=False):
            x = np.array(x)
            if return_randperm: 
                randperm = torch.randperm(len(x))
                return list(map(lambda x:str(x), list(x[randperm]))), randperm
            return list(map(lambda x:str(x), list(x[randperm])))
        def scramble_and_get_answer(example):
            scramble_list_choices, randperm = scramble_list(example["mc1_targets"]["choices"], return_randperm=True)    
            problem_str = example["question"] + " \n " \
                        + "\n ".join([f"{num_to_chr(i)} {opt}" for i, opt in enumerate(scramble_list_choices)])
            permutation = torch.tensor(example["mc1_targets"]["labels"])[randperm]
            i = torch.where(permutation==1)[0]

            return {"problem":problem_str, "formatted_answer":f"{num_to_chr(i)}"}
        dataset = dataset.map(lambda example: {**example, **scramble_and_get_answer(example)})
    elif args.dataset == "competition_math":
        dataset = load_dataset(args.dataset,split=args.split,trust_remote_code=True)
        def extract_comp_math_question(str, pattern=r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", default=None):
            # Applying the regex
            match = re.search(pattern, str)
            # Extract the content
            return match.group(1) if match else None
        dataset = dataset.map(lambda example:{**example,"formatted_answer":extract_comp_math_question(example["solution"])})
    elif args.dataset == "cais/mmlu":
        dataset = load_dataset("cais/mmlu", "all",split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(example["choices"])]),
                    "formatted_answer": num_to_chr(example["answer"])
                    }
                )
    elif args.dataset in ["allenai/ai2_arc@ARC-Challenge", "allenai/ai2_arc@ARC-Easy"]:
        dataset_name, dataset_config = args.dataset.split("@")
        dataset = load_dataset(dataset_name, dataset_config,split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join([f"({label}) {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])]),
                    "formatted_answer": add_parans(example["answerKey"])
                    },
                )
    elif args.dataset == "deepmind/aqua_rat":
        dataset = load_dataset(args.dataset,"raw", split=args.split)
        dataset = dataset.map(lambda example: {
                    **example,
                    "problem": example["question"] + "\n " \
                        + "\n ".join(example["options"]),
                    "formatted_answer": add_parans(example["correct"])
                    }
                )
    else:
        assert False, "Other datasets not supported"
    
    dataset = dataset.select_columns(["problem","formatted_answer"]).shuffle()
    if args.num_samples:
        dataset=dataset.select(range(min(args.num_samples, len(dataset))))
    return dataset


def pack_tokens(pipeline, question_tokens):
    max_len         = max([len(inp) for inp in question_tokens])
    input_ids       = [[pipeline.tokenizer.eos_token_id for _ in range(max_len-len(inp))] + inp for inp in question_tokens]
    attention_mask  = [[0 for _ in range(max_len-len(inp))] + [1 for _ in range(len(inp))] for inp in question_tokens]
    inputs = {
        "input_ids":torch.tensor(input_ids),
        "attention_mask":torch.tensor(attention_mask)
    }
    return inputs

def generate_tokens(pipeline, question_tokens, prompt_gen, max_gen_length):
    inputs = pack_tokens(pipeline, question_tokens)
    with torch.no_grad():
        outputs = pipeline.model.generate(input_ids=inputs["input_ids"].long().to(device), 
                                            attention_mask=inputs["attention_mask"].to(device),
                                            max_new_tokens=max_gen_length)
        outputs = outputs.cpu()
    responses = []
    outputs_ = [] 
    for i in range(outputs.shape[0]):
        begin_index = torch.where(outputs[i]==prompt_gen.heading_to_tokens["begin_of_text"])[0][0]
        end_where = list(torch.where(outputs[i]==prompt_gen.heading_to_tokens["<|end_of_text|>"])[0])
        end_index = end_where[0] if len(end_where) > 0 else len(outputs[i])
        outputs_.append(outputs[i][begin_index:end_index].cpu())
        responses.append(pipeline.tokenizer.decode(outputs_[-1], skip_special_tokens=False))
    return inputs, outputs_, responses

def create_dataframe(data, columns):
    extracted_data = {col: [d[col] for d in data] for col in columns}
    df = pd.DataFrame(extracted_data)
    return df

## https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
def ceildiv(a, b): 
    return -(a // -b)

