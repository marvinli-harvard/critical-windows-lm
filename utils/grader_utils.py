import requests
import re
from enum import Enum
from tqdm import tqdm
from typing import List 

import torch
import transformers

from vllm import LLM, SamplingParams
from prm800k.grading.grader import grade_answer

from utils.configuration import *

## Chain of Thought Q&A graders
def extract_answer(input: str, type_answer: str) -> str:
    """
    Extracts the answer from the given input string based on the specified type of answer.
    Args:
        input (str): The input string containing the answer.
        type_answer (str): The type of answer to extract. Can be either "multiple_choice" or "math".
    Returns:
        str: The extracted answer. For multiple choice, the answer is returned in the format (A), (B), etc.

             For math, the answer is extracted.
    Raises:
        ValueError: If an unsupported type_answer is provided.
    Notes:
        - For "multiple_choice", the function attempts to match various patterns to extract the answer.
        - For "math", the function attempts to match various patterns to extract the mathematical expression.
        - If no valid answer is found, an empty string is returned and a message is printed.
    """
    if type_answer == AnswerType.MULTIPLE_CHOICE.value:
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
            re.compile(r'answer\s*-\s*\(?([A-Za-z])\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*\(?(\d+)\)?', re.IGNORECASE),  # Matches (2), 4
            re.compile(r'the answer is\s*\(?(\d+)\)?\s+\w+', re.IGNORECASE),  # Matches (1) community, (4) bacteria
            re.compile(r'answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?', re.IGNORECASE),
            re.compile(r'answer:\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?', re.IGNORECASE),
            re.compile(r'the answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?', re.IGNORECASE),
            re.compile(r'correct answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?', re.IGNORECASE),
            re.compile(r'option\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*option\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'answer is\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'answer:\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'correct answer is\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'answer\s*[=:]\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'answer\s*-\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*\(?(\d+)\)?', re.IGNORECASE),
            re.compile(r'the answer is\s*\(?(\d+)\)?\s+\w+', re.IGNORECASE),
            re.compile(r'the answer is ([A-Za-z0-9]+)\)?', re.IGNORECASE),
        ]

        # Attempt to find matches for each pattern
        for pattern in patterns:
            match = pattern.search(input)
            if match:
                return f"({match.group(1).upper()})"  # Normalize the format as (A), (B), etc.

        # If no patterns match, print
        print(f"No valid multiple-choice answer found in input: {input}")
        return ""
    
    elif type_answer == AnswerType.MATH.value:
        # Match phrases like 'answer is 42', 'answer: 3.14', etc.
        patterns = [
                re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.IGNORECASE),
                re.compile(r'the answer is\s*\$\\boxed{([^}]+)}\$', re.IGNORECASE),  # Matches $\boxed{EXPRESSION}$
                re.compile(r'the answer is\s*\$\\boxed{([^}]+)}\$', re.IGNORECASE),  # Matches $\boxed{EXPRESSION}$
                re.compile(r'correct answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?',re.IGNORECASE),
                re.compile(r'correct answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'answer:\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'answer:\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'the answer is ([A-Za-z0-9]+)\)?', re.IGNORECASE),
                re.compile(r'the answer is\s*\$?\\boxed{\(?([A-Za-z0-9])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'the answer is\s*\$?\\boxed{\(?([A-Za-z])\)?}\$?',
                            re.IGNORECASE),
                re.compile(r'the answer is\s*\$([^$]+)\$', re.IGNORECASE),  # Matches $EXPRESSION$
                re.compile(r'the answer is\s*([^\.]+)\.', re.IGNORECASE),  # Matches EXPRESSION.
                re.compile(r'answer:\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'correct answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'answer:\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'the answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'answer is\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'answer\s*-\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'answer\s*-\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'the answer is\s*option\s*\(?([A-Za-z0-9])\)?',
                            re.IGNORECASE),
                re.compile(r'option\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'answer\s*[=:]\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'the answer is\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'answer\s*[=:]\s*\(?([A-Za-z])\)?', re.IGNORECASE),
                re.compile(r'correct answer is\s*\(?([A-Za-z0-9])\)?',
                            re.IGNORECASE),
                re.compile(r'the answer is\s*option\s*\(?([A-Za-z])\)?',
                            re.IGNORECASE),
                re.compile(r'answer is\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'option\s*\(?([A-Za-z0-9])\)?', re.IGNORECASE),
                re.compile(r'the answer is\s*\(?(\d+)\)?\s+\w+', re.IGNORECASE),
                re.compile(r'the answer is\s*\(?(\d+)\)?', re.IGNORECASE)
        ]
                        

        # Attempt to find matches for each pattern
        for pattern in patterns:
            match = pattern.search(input)
            if match:
                return match.group(1)  # Return the mathematical answer as is

        # If no patterns match, print
        print(f"No valid math answer found in input: {input}")
        return ""
    
    else:
        # If an unsupported type_answer is provided, raise an error
        print(f"Unsupported answer type: {type_answer}")
        return ""

def compare_answers(answer1: str, answer2: str, type_answer: str) -> bool:
    """
    Compares two answers based on the type of answer.
    Args:
        answer1 (str): The first answer to compare.
        answer2 (str): The second answer to compare.
        type_answer (str): The type of answer, which can be "multiple_choice" or "math".
    Returns:
        bool: True if the answers are considered equal based on the type, False otherwise.
    Raises:
        ValueError: If the type_answer is not supported.
    """

    if type_answer == "multiple_choice":
        return answer1.strip().upper() == answer2.strip().upper()
    elif type_answer == "math":
        return grade_answer(answer1, answer2)
    else:
        raise ValueError(f"Unsupported answer type: {type_answer}")

## Jailbreak classification
class JailBreakClassifierWrapper:
    def __init__(self, batch_size : int = 4):
        self.judge_templates = requests.get("https://raw.githubusercontent.com/dsbowen/strong_reject/main/strong_reject/eval_files/judge_templates.json").json()
        self.model_name = "qylu4156/strongreject-15k-v1"
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=bnb_config
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left", truncation_side="left"
        )
        self.batch_size = batch_size
    
    def grade_answers(self, 
                      prompts: List[str], 
                      generations: List[str],
                      max_response_length=512
                      ):
        truncated_responses = []
        for response in generations:
            truncated_response = self.tokenizer(response, max_length=max_response_length, truncation=True)[
                "input_ids"
            ]
            truncated_responses.append(self.tokenizer.decode(truncated_response))

        judge_prompts = [
            self.judge_templates["strongreject_finetuned"].format(
                forbidden_prompt=forbidden_prompt, response=response
            )
            for forbidden_prompt, response in zip(prompts, truncated_responses)
        ]
        tokenized_judge_prompts = self.tokenizer(judge_prompts, padding=True, return_tensors="pt")
        all_scores = []
        for i in range(0, len(judge_prompts), self.batch_size):
            logits = self.model(
                input_ids=tokenized_judge_prompts["input_ids"][i:i+self.batch_size,].to(self.model.device),
                attention_mask=tokenized_judge_prompts["attention_mask"][i:i+self.batch_size,].to(self.model.device),
            ).logits[:, -1]
            possible_scores = "1", "2", "3", "4", "5"
            probs = logits[:, [self.tokenizer.vocab[i] for i in possible_scores]].softmax(dim=-1)
            all_scores.extend((probs * torch.linspace(0, 1, 5, device=self.model.device)).sum(dim=-1).tolist())
        return all_scores
