"""
utils.py

Some basic utilities
"""
import os
from tqdm import tqdm
import re
import math
from typing import Optional, List

import pandas as pd
import numpy as np

from utils.configuration import *



# String and Arithmetic Manipulations
def ceildiv(a : int , b : int) -> int: 
    ## https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
    return -(a // -b)

def num_to_chr(x : int) -> str:
    return f"({chr(65+x)})"

def add_parans(x : int) -> str:
    return f"({x})"

## String manipulation
def extract_first_assistant_response(text :str) -> Optional[str]:
    """"Extract first assistant response"""
    # Regular expression to find all assistant sections
    matches = [
        re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", text, re.DOTALL),
        re.findall(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)", text, re.DOTALL)
    ]
    # If matches are found, return the last one, stripping any extra whitespace
    for m in matches:
        if m:
            return m[0]
    return None  # Return None if no matches are found

def extract_comp_math_question(string : str, pattern=r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"):
    """"Extract answer from MATH competition question"""
    # Applying the regex
    match = re.search(pattern, string)
    # Extract the content
    return match.group(1) if match else None

def longest_common_prefix(str1 :str, str2:str) -> str:
    """Finds the longest common prefix shared between two strings using os.path.commonprefix."""
    return os.path.commonprefix([str1, str2])

def get_prefix(problem_rows:pd.DataFrame, orig_string_ans : str, ind : float) -> str:
    """
    Extracts the longest common prefix between the original string answer and the stump string answer
    for a given index from the problem rows DataFrame.
    Args:
        problem_rows (pd.DataFrame): DataFrame containing problem rows with 'stop_frac' and 'stump_string_ans' columns.
        orig_string_ans (str): The original string answer to compare against.
        ind (float): The index used to filter the problem rows DataFrame.
    Returns:
        str: The longest common prefix between the original string answer and the stump string answer,
                split at the "<|eot_id|>" marker if present.
    """
    stump_ans  = problem_rows[problem_rows.stop_frac==ind]["stump_string_ans"].unique()[0].split("Let\'s think step by step:")[1]
    return longest_common_prefix(orig_string_ans,stump_ans).split("<\|eot_id\|>")[0]

def highlight_difference(orig_string_ans : str, before_str : str, after_str : str) -> str:
    """
    Highlights the difference between two substrings within an original string.
    This function takes an original string and two substrings, and highlights the part of the original string
    that differs between the two substrings by wrapping it in a span with a yellow background color.
    Args:
        orig_string_ans (str): The original string containing the full text.
        before_str (str): The substring representing the text before the change.
        after_str (str): The substring representing the text after the change.
    Returns:
        str: The original string with the differing part highlighted in yellow.
    """
    # Determine the lenfth of the prefix 
    prefix = len(before_str)
    # Determine where `after_str` ends in the original string
    last_shared = len(after_str)

    # Construct the HTML string with highlighted text
    return (
        f"{orig_string_ans[:prefix]}"  # Text before the highlighted part
        + f"<span style='background-color: yellow;'>{orig_string_ans[prefix:last_shared]}</span>"  # Highlighted in-between part
        + f"{orig_string_ans[last_shared:]}"  # Text after the highlighted part
    )


def save_dataframe_as_html(df:pd.DataFrame)->str:
    """
    Generates an HTML file from the DataFrame with a pretty layout for each row.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data to display.
    Returns:
        str: html of string
    """
    rows_html = []
    for _, row in df.iterrows():
        # Construct the HTML content for each row
        row_html = f"""
        <div style="border: 1px solid #ddd; margin: 10px; padding: 10px; font-family: Arial, sans-serif;">
            <strong>Dataset Source:</strong> {row['dataset_source']}<br>
            <strong>Problem:</strong> {row['problem']}<br>
            <strong>Id:</strong> {row['id']}<br>
            <strong>Correct Answer:</strong> {row['formatted_answer']}<br>
            <strong>Original Answer:</strong> {row['orig_ans_format']}<br>
            <strong>stop_frac:</strong> {row['stop_frac']}<br>
            <strong>Highlighted Difference:</strong> {row['highlighted_diff']}
        </div>
        """
        rows_html.append(row_html)

    # Combine all rows into a single HTML page
    html_content = f"""
    <html>
    <head>
        <title>Highlighted Differences</title>
    </head>
    <body>
        <h1 style="font-family: Arial, sans-serif; text-align: center;">Highlighted Differences</h1>
        {''.join(rows_html)}
    </body>
    </html>
    """
    
    # Save the HTML content to a file
    return html_content

## DataFrame utils
def percent_to_freq(percent:float, synthetic_df: pd.DataFrame, final_str :str, default_column:str="asst_response"):
    assert 0<=percent<=1
    prefix = final_str[:int(len(final_str)*percent)]
    return (synthetic_df.loc[synthetic_df[default_column].str.startswith(prefix),default_column]==final_str).mean()

def total_variation_distance(list1, list2):
    """
    Compute the total variation distance between two empirical distributions.
    
    Args:
        list1 (list or numpy array): First list of values.
        list2 (list or numpy array): Second list of values.
    
    Returns:
        float: The total variation distance.
    """
    # Combine both lists and get unique elements
    unique_values = np.unique(np.concatenate((list1, list2)))
    
    # Compute empirical probabilities
    prob1 = np.array([list1.count(val) / len(list1) for val in unique_values])
    prob2 = np.array([list2.count(val) / len(list2) for val in unique_values])
    
    # Calculate total variation distance
    tv_distance = 0.5 * np.sum(np.abs(prob1 - prob2))
    
    return tv_distance

def load_df_across_dirs(datasets : List[str], dataset_names : List[str], base_dir = str, 
            combine_str : Optional[str] = "dataset_with_gens_noisedenoise_ans.csv",
            drop_columns : List[str] = ["formatted_answer"])->pd.DataFrame:
    combined_df = pd.DataFrame()

    for dataset, dataset_name in tqdm(list(zip(datasets, dataset_names))):
        file_path = os.path.join(base_dir, dataset, combine_str)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['dataset_source'] = dataset_name
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    for col in drop_columns:
        print(f"Before dropping NA for {col} : length = {len(combined_df)}")
        combined_df = combined_df[~combined_df[col].isna()]
        print(f"After dropping NA for {col} : length = {len(combined_df)}")
    
    return combined_df 

def cw_condition(y_diffs : np.array, cw : float=0.5, mon : float =-0.3):
    """
    Check if the given differences meet the critical window condition.
    This function evaluates whether the differences in the input array `y_diffs`
    satisfy two conditions:
    1. There is at least one difference greater than the critical window threshold `cw`.
    2. All differences are greater than or equal to the monotonic threshold `mon`.
    
    Args:
        y_diffs (array-like): An array of differences to be evaluated.
        cw (float, optional): The critical window threshold. Default is 0.5.
        mon (float, optional): The monotonic threshold. Default is -0.3.
    Returns:
        bool: True if both conditions are met, False otherwise.
    """

    has_large_jump = np.any(y_diffs > cw)
    is_monotonic = np.all(y_diffs >= mon)
    return has_large_jump and is_monotonic

def extract_synthetic_choices(template_string, input_string):
    # Step 1: Extract all options from the template string
    template_options = re.findall(r"\((\w+)/(\w+)\)", template_string)
    
    # Step 2: Extract sentences from the input string
    input_sentences = re.findall(r"\d+\.\s(.*?)(?:\n|$)", input_string)
    
    # Step 3: Match chosen words (case-insensitive) in order
    chosen_words = []
    for options, sentence in zip(template_options, input_sentences):
        # Convert to lowercase for comparison
        word1, word2 = options[0].lower(), options[1].lower()
        sentence_lower = sentence.lower()
        
        # Check which of the two options is present in the sentence
        if word1 in sentence_lower:
            chosen_words.append(options[0])  # Append original-cased word
        elif word2 in sentence_lower:
            chosen_words.append(options[1])  # Append original-cased word
    
    return chosen_words

def find_normalized_positions(template_str: str, final_str: str, step_size: float = 0.01):
    """
    Finds the normalized positions of chosen words in a string, with adjustable step size for rounding.
    Handles case-insensitive matching robustly.

    Args:
        template_str (str): The template string containing word options (option1/option2).
        final_str (str): The final string with chosen words.
        step_size (float): The step size for rounding (e.g., 0.01, 0.05, 0.1).
    
    Returns:
        tuple: tlower_points (rounded down), tupper_points (rounded up).
    """
    # Step 1: Extract words inside parentheses in the template string
    options = re.findall(r'\((\w+)/(\w+)\)', template_str)
    chosen_words = []

    # Step 2: Perform case-insensitive matching
    lower_final_str = final_str.lower()  # Convert the final string to lowercase for matching
    
    for option1, option2 in options:
        if option1.lower() in lower_final_str:  # Check for lowercase match
            chosen_words.append(option1)
        elif option2.lower() in lower_final_str:
            chosen_words.append(option2)

    # Step 3: Find positions of the first letter of each chosen word (case-sensitive)
    positions = [final_str.lower().find(word.lower()) for word in chosen_words]
    
    # Step 4: Normalize positions based on the total string length
    total_length = len(final_str)
    scale = 1 / step_size  # Scaling factor for adjustable rounding

    tlower_points = [math.floor(pos / total_length * scale) / scale for pos in positions]
    tupper_points = [math.ceil((pos + 1) / total_length * scale) / scale for pos in positions]

    return tlower_points, tupper_points

def match_rows_with_precision(df, target_values, precision=1e-6):
    """
    Match float-based indices of a DataFrame to target values within a specified precision.

    Parameters:
        df (pd.DataFrame): DataFrame with a float-based index.
        target_values (array-like): List or array of target values to match.
        precision (float): Acceptable precision threshold for matching.

    Returns:
        pd.DataFrame: Subset of the original DataFrame with matched rows.
    """
    # Convert index and target values to NumPy arrays
    index_array = df.index.to_numpy()
    target_values_array = np.array(target_values)

    # Calculate absolute differences using broadcasting
    abs_diff = np.abs(index_array[:, None] - target_values_array)
    mask = abs_diff <= precision  # Mask where differences are within precision

    # Find indices that have valid matches
    matched_indices = index_array[np.any(mask, axis=1)]

    # Return the matched rows
    return df.loc[matched_indices]

