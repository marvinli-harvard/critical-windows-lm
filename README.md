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


## To-dos
- [ ] Explore critical windows CoT and come up with some sort of explanation/hypothesis
- [ ] Construct dataset with different jailbreaks and create critical windows for jailbreaking\
        - [ ] Plot critical windows for jailbreaks
- [ ] Create new persona token and finetune language model on preference data to always output it

## Experiments

### CoT descriptive experiments
- [ ] Overall percentages (reminder to reproduce because shuffle was not applied for all but one dataset)\
        - [ ] Compare with just asking the model for the answers \
- [ ] Run curves for 400 examples from each dataset\
        - [ ] Look at specific examples and find some sort of pattern

### Jailbreaking descriptive experiments
- [ ] Generate critical windows for jailbroken prompts

### Monitoring with likelihood ratio
- [ ] Use Jailbroken llama model for monitoring jailbreaks\
        - [ ] Identifying jailbreaks after they are completed\
        - [ ] Identifying jailbreaks from only the prompt\
        - [ ] Identifying jailbreaks in the middle\
        - [ ] Envisioning scatterplot between likelihood ratio and prob jailbreak 

### Finetuning
- [ ] Use finetuning to predict prob of answering earlier accurately\
        - [ ] to predict prob of being jailbroken\
- [ ] Finetuning LLM to reflect/remind itself of its purpose after every k tokens, see if that breaks off-distribution behavior
- [ ] Finetune a language model to only produce a particular answer

### Monitoring with interp
- [ ] Use interp to predict\
        - prob of jailbreak succeeding\
        - prob of certain multiple choice answer
