from prm800k.grading.grader import grade_answer
import re
from enum import Enum

class AnswerType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    MATH = "math"

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
