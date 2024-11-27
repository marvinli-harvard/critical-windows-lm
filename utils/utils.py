"""
utils.py

Some basic utilities
"""
import os
import re
from typing import Optional

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

## String manipualtion
def extract_first_assistant_response(text :str) -> Optional[str]:
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
