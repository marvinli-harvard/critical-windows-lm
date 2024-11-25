## Useful prompts
SYSTEM_PROMPT = "Produce a correct solution to the following /TASK/ question."
COT_PROMPT = \
    "Think of the /TASK/ question thoroughly step by step. \
    Please only respond with the answer after reasoning thoroughly. \n\n"

CLARIFY_CHOICE_STR_MC = "Given all of the above, what’s the single, most likely answer? Your answer should have the format \"The answer is ANSWER\", where ANSWER is your answer. \n\n"
CLARIFY_CHOICE_STR_MATH = "Given all of the above, what’s the single, most likely answer? Simplify it completely. Your answer should have the format \"The answer is $ANSWER$\", where ANSWER is your answer in LaTeX. \n\n"

## List of words in repeat word dataset (from Carlini et al. 2024)
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

## Constants for generation
DEFAULT_SEED = 2243
MAX_GEN_LEN_ANSWER = 100 
MAX_GEN_LEN_COT = 2048
DEFAULT_PERCENT_PROMPT = [0.05, 0.1,0.3,0.5,0.7,0.9]
