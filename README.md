# critical-windows-lm

In this repo, we explore the phenomena of critical windows in the context of language models

Datasets to demonstrate this phenomena with chain of thought (multiple choice questions)
1. LogiQA - DONE
2. TruthfulQA - DONE
3. MATH - DONE
4. ARC Challenge - DONE
5. ARC Easy - DONE
6. MMLU - DONE

Need to run the following code to get `grade_answer` function (from MATS interview)
```python
!git clone https://github.com/openai/prm800k.git

import sys
from google.colab import files
%cd /content/prm800k
!pip install -e .
!pip install pylatexenc

with open('/content/prm800k/prm800k/grading/grader.py', 'r') as file:
    content = file.read()

# Make a small modification to handle relative imports
modified_content = content.replace(
    'from grading import math_normalize',
    'from . import math_normalize'
)

# Write back to the file
with open('/content/prm800k/prm800k/grading/grader.py', 'w') as file:
    file.write(modified_content)

from prm800k.grading.grader import grade_answer
```
In addition to ``environment.yml``, we need to install 
```bash
pip install vllm
```

## DONE
- [X] Implement AquA
- [X] Include correctness information
- [X] Batchify noise denoise
- [X] Make eval to compare with true answer & work with batch
    -   Check that new graders reduce number of Nones
- [X] Run on 10 individual samples with lots of data from each noising and denoising time level
- [X] Convert everything to vllm
- [X] Write code to compare with asking for answer directly
- [X] Run on 400 samples with lots of data from each noising and denoising time level
- [X] Make `is_stump` with `is_consistent` plot
- [X] Overall diagrams for 10k dataset
- [X] Refactor some code
- [X] Explore critical windows CoT and come up with some sort of explanation/hypothesis - Important parts of the reasoning process

## To-dos
- [ ] Construct dataset with different jailbreaks and plot critical windows for jailbreaking\
- [ ] Finetune a language model to emphasize higher probabilty sections 

## Experiments

### Descriptive experiments
- [X] Overall percentages (reminder to reproduce because shuffle was not applied for all but one dataset)\
        - [X] Compare with just asking the model for the answers \
- [X] Run curves for 400 examples from each dataset\
        - [X] Look at specific examples and find some sort of pattern
- [ ] Generate critical windows for jailbreaks
- [ ] Generate critical windows for synthetic data

### Methods/prescription to try
- [ ] Likelihood ratio between jailbroken and not jailbroken model to predict prob of jailbreak behavior
- [ ] Finetuning LLM with higher weight on important tokens
- [ ] Mech Interp to predict probability of jailbreaks 
- [ ] Finetuning LLM to reflect/remind itself of its purpose after every k tokens, see if that breaks off-distribution behavior

## Less promising ideas
### Finetuning
- [ ] Finetune a language model to predict probability of its future predictions

### Monitoring 
- [ ] Finetune a language model to only produce a particular answer and use that to predict prob of predicting answer\
- [ ] Use interp to predict prob of certain answer
