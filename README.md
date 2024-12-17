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


## Reading to-dos
- [ ] Theory for Bayesian linear regression
- [ ] Read phi paper and bailey paper

### Coding modifications/re-running
- [ ] Move everything from jupyter notebook to scripts
- [ ] Run experiments on Gemini-7b instruct

### Improve rigor of jailbreak experiments
- [ ] Switch to using StrongReject as auditor for jailbreaks
- [ ] Use dataset from Luke Bailey paper to compute jailbreak accuracy for prompts frequently confused as jailbroken
- [ ] switch to testing more jailbreak types and looking at JailbreakBencg
- [ ] baselines for jailbreaks (probes or text classifiers)
- [ ] Switch from jailbroken to pre-instruct model

## Experiments

### Descriptive experiments
- [X] Overall percentages (reminder to reproduce because shuffle was not applied for all but one dataset)\
        - [X] Compare with just asking the model for the answers \
- [X] Run curves for 400 examples from each dataset\
        - [X] Look at specific examples and find some sort of pattern
- [X] Generate critical windows for jailbreaks 
- [X] Generate critical windows for synthetic data 

### Methods/prescription to try
- [x] Likelihood ratio between jailbroken and not jailbroken model to predict prob of jailbreak behavior
- [x] See if prompting LLM with critical windows makes it easier to correct its mistakes

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
- [X] Construct dataset with different jailbreaks and plot critical windows for jailbreaking\
- [X] Synthetic data 
- [X] See if reminding LLM of critical windows makes it better able to correct
- [X] Likelihood ratio between jailbroken and not jailbroken model to predict prob of jailbreak behavior
- [X] Fix increased size of dataset from 1k to 1.008k during merge of `instruction` of aligned data for GPT
- [x] get rid of madlib wording because it isnt accurate 
- [x] plot frequency of jailbreaks/info regarding jumps 

