"""
Copyright: anonymous
Time: 2024.11.29
Function: data processors for distillating prejudge and critique traininfg examples.
"""

import sys
sys.path.append("./")
sys.path.append("../")
import os
import json
import re
from tqdm import tqdm
import random
from typing import Dict
from utils.answer_extractor import last_boxed_only_string, remove_boxed, strip_string

from examples.config.data_params import DISTILLATING_PARAMS
from processors.base import BaseProcessor

### 构建用于prejudge generation、critique revise的prompt
TEMPLATE = """You possess expertise in solving mathematical problems through a systematic, step-by-step reasoning process during which you are dedicated to preventing repeating any errors analyzed in experiences. Here is a problem and the corresponding correct answer:

Problem: 
```
{Problem}
```
Correct Answers: 
```
{Correct_Answer}
```

Now, I will give you the initialized reasoning solution steps, a correcrt completion, and some corresponding incorrect completions which aim at continual finishing the rest of the solution steps but reach the wrong answer. The reasoning situations are in the following:

Initialized Reasoning Steps: 
```
{Prefix_Response}
```

Correct Completion:
```
{Suffix_Correct_Response}
```

{Suffix_Incorrect_Responses}

Please help me and give some following tips:
1) Misconception: Each incorrect completion reach incorrect answer due to misconception, please list the candidate misconceptions. Note: Please analyze and list the possible errors directly, and DO NOT disclose the complete number (e.g., "Completion #1") and information.
2) Critique: Suppose that you fall into one of the incorrect reasoning step, please generate a critique with error analysis as the error experience. The output format style can be one of but not limited to "Wait, let me verify the current step, ...", "Ooops, I think I could make some misconceptions, ...", "May be I was wrong, ...", etc.
3) Prejudge: Suppose that you solve this problem when you start reasoning from the give initialized step, please generate a prejudge information to ask yourself to avoid the misconception. Note: The generated prejudge should NOT be general, and MUST closely integrate with the problem and the initilized reason step. The output format style can be one of but not limited to "Wait, I need to be careful about the rest solution step, ...", "Oh, I think the next solution may make some misconceptions, let me prejudge before finish the rest, ...", etc.

Please note that the final output format must be in the following template:
### Misconception
...
### Critique
...
### Prejudge
...
"""

Incorrect_Suffix_Template = """Incorrect Completion #{Number}:
```
{Suffix_Incorrect_Response}
```
"""

class PrejudgeCritiqueProcessor(BaseProcessor):
    def __init__(self, data_params: dict):
        self.data_params = data_params
        self.data_name = self.data_params["data_name"]
        self.train_examples, self.dev_examples, self.test_examples = list(), list(), list()
        self.template_hub: Dict[str, str] = {
            "template": TEMPLATE,
            "incorrect_suffix_template": Incorrect_Suffix_Template,
        }

    def prompting(self, examples):
        for ei, example in enumerate(tqdm(examples)):
            example["input"] = self.__prompting__(example)
            if ei == 0:
                print(example["input"])
        return examples

    def __prompting__(self, example):
        TEMPLATE = self.template_hub["template"]
        Incorrect_Suffix_Template = self.template_hub["incorrect_suffix_template"]
        
        suffix_responses = example["suffix_responses"]
        suffix_responses_rewards = example["suffix_responses_rewards"]
        incorrect_suffix_responses = [suffix_response for suffix_response, suffix_reward in zip(suffix_responses, suffix_responses_rewards) if suffix_reward == 0]
        correct_suffix_responses = [suffix_response for suffix_response, suffix_reward in zip(suffix_responses, suffix_responses_rewards) if suffix_reward == 1]

        
        incorrect_suffix_prompt = ""
        if len(incorrect_suffix_responses) > 4:
            random.shuffle(incorrect_suffix_responses)
        incorrect_suffix_responses = incorrect_suffix_responses[:4]
        for ei, response in enumerate(incorrect_suffix_responses):
            incorrect_suffix_prompt += Incorrect_Suffix_Template.format(Number=ei+1, Suffix_Incorrect_Response=response)
        incorrect_suffix_prompt = incorrect_suffix_prompt.strip()

        prompt = TEMPLATE.format(
            Problem=example["prompt"],
            Correct_Answer=example["labels"][0],
            Prefix_Response=example["prefix_response"],
            Suffix_Correct_Response=correct_suffix_responses[0],
            Suffix_Incorrect_Responses=incorrect_suffix_prompt,
        )
        return prompt

