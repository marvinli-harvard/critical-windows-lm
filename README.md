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

## To-dos 
- [x] Implement AquA
- [x] Include correctness information
- [x] Batchify noise denoise
- [ ] Make eval to compare with true answer & work with batch
    -   Check that new graders reduce number of Nones
- [ ] Run on 10 individual samples with lots of data from each noising and denoising time level
- [ ] Construct dataset with difference jailbreaks and compute probability of jailbreak occuring 
- [ ] Create new persona token and finetune language model on preference data to always output it

## Experiments
### CoT descriptive experiments
- [ ] Overall percentages (reminder to reproduce because shuffle was not applied for all but one dataset)
- [ ] Run curves for 10x examples from each dataset

### Jailbreaking descriptive experiments
- [ ] Generate critical windows examples for jailbroken prompts

### In-context learning descriptive experiments (?)
- [ ] Generate critical windows examples for in-context examples

### Monitoring with likelihood ratio
- [ ] Use Jailbroken llama model for monitoring jailbreaks\
        - Identifying jailbreaks after they are completed\
        - Identifying jailbreaks from only the prompt\
        - Identifying jailbreaks in the middle\
        - Envisioning scatterplot between likelihood ratio and prob jailbreak 
- [ ] Finetune a language model to only produce a particular answer

## Monitoring with interp
- [ ] Use interp to predict\
        - prob of jailbreak succeeding\
        - prob of certain multiple choice answer

## Finetuning to print a special token
- [ ] Finetuning LLM to reflect/remind itself of its purpose after every k tokens, see if that breaks off-distribution behavior

