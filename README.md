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
