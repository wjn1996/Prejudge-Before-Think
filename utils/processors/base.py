"""
Copyright: anonymous
Time: 2024.11.29
Function: data processors for reasoning examples.
"""

import sys
sys.path.append("./")
sys.path.append("../")
import os
import json
import re
from tqdm import tqdm
from typing import Dict
from utils.answer_extractor import last_boxed_only_string, remove_boxed, strip_string

from examples.config.data_params import BOOTSTRAPPING_PARAMS

class BaseProcessor:
    def __init__(self, data_params: dict):
        self.data_params = data_params
        self.data_name = self.data_params["data_name"]
        self.train_examples, self.dev_examples, self.test_examples = list(), list(), list()
        self.template_hub: Dict[str, str] = {
            "io": "{Question}",
            "ao": "{Question}\nPlease only output the final answer in \\boxed.",
            "zeroshot_cot": "{Question}\nLet's think step by step, and then output the final answer in \\boxed.",
            "zeroshot_step_cot": "{Question}\nLet's think step by step, each step must split by the token <|reserved_special_token_0|>, then output the final answer in \\boxed.",
            "fewshot_step_cot": "Q: The first four terms in an arithmetic sequence are $x+y$, $x-y$, $xy$, and $x/y$, in that order. What is the fifth term?\nA: To find the fifth term, I need to identify the common difference of the arithmetic sequence and add it to the fourth term.<|reserved_special_token_0|>The common difference is the same for any consecutive pair of terms, so I can use any of them to find it.<|reserved_special_token_0|>For example, using the first and second terms, I can write $x-y = x+y + d$, where $d$ is the common difference.<|reserved_special_token_0|>Solving for $d$, I get $d = -2y$.<|reserved_special_token_0|>Using another pair of terms, such as the second and third, I can check if this value of $d$ is consistent.<|reserved_special_token_0|>I have $xy = x-y + d$, so substituting $d = -2y$, I get $xy = x-y - 2y$.<|reserved_special_token_0|>Simplifying, I get $xy = x - 3y$.<|reserved_special_token_0|>This seems like a reasonable equation, so I will assume that $d = -2y$ is correct.<|reserved_special_token_0|>Now, to find the fifth term, I need to add $d$ to the fourth term.<|reserved_special_token_0|>The fourth term is $x/y$, so the fifth term is $x/y + d = x/y - 2y$.<|reserved_special_token_0|>To express this as a common fraction, I need to find a common denominator for $x/y$ and $-2y$.<|reserved_special_token_0|>The least common denominator is $y$, so I can multiply the numerator and denominator of $-2y$ by $y$ to get $-2y^2/y$.<|reserved_special_token_0|>Therefore, the final answer is \\boxed{$x/y - 2y^2/y = (x - 2y^2)/y$}.<|reserved_special_token_0|>\nQ: {Question} Let's think step by step, each step must split by the token <|reserved_special_token_0|>, then output the final answer in \\boxed.\nA: ",
            "answer_format_cot": "{Question}\nLet's think step by step, and then output the final answer in \\boxed. Please note that your final answer should has the similar format to {Format}.",
            "answer_format_step_cot": "{Question}\nLet's think step by step, each step must split by the token <|reserved_special_token_0|>, then output the final answer in \\boxed. The answer format is very same to \"{Format}\".",
        }

    def load_data(self, data_kind: str = "train"):
        examples = self.__read_file__(self.data_params["data_path"][data_kind])
        if data_kind == "train":
            self.train_examples = examples
        elif data_kind == "dev":
            self.dev_examples = examples
        elif data_kind == "test":
            self.test_examples = examples

        print("[{} data num: {}]".format(data_kind, len(examples)))
    
    def get_data(self, data_kind: str = "train"):
        if data_kind == "train":
            return self.train_examples
        elif data_kind == "dev":
            return self.dev_examples
        elif data_kind == "test":
            return self.test_examples
        return None
    
    def __read_file__(self, data_path):
        print("reading {} examples ...".format(self.data_name))
        examples = list()
        with open(data_path, "r", encoding="utf-8") as fr:
            examples = [json.loads(line) for line in tqdm(fr.readlines())]
        return examples

    def prompting(self, examples, prompt_kind):
        for example in tqdm(examples):
            example["input"] = self.__prompting__(example, prompt_kind)
        return examples

    def __prompting__(self, example, prompt_kind):
        template = self.template_hub[prompt_kind]
        if "{Question}" in template and "question" in example.keys():
            template = template.replace("{Question}", example["question"])
        if "Format" in template and "answer_format" in example.keys():
            template = template.replace("{Format}", example["answer_format"])
        return template


    @classmethod
    def extract_answer(cls, text: str):
        """
        Args:
            text (str): 将要抽取的文本
        Return:
            (any): 抽取的结果
        """
        extract_ans = text.strip()

        # 如果有boxed，优先按照boxed抽取
        if last_boxed_only_string(extract_ans):
            extract_ans = remove_boxed(last_boxed_only_string(extract_ans))
        # 如果有The answer is:， 优先按照The answer is:抽取
        elif re.search(r"The answer is:(.+?)$", text):
            extract_ans = re.search(r"The answer is:(.+?)$", text).group(1)
        elif re.search(r"<Answer_Start>(.+?)<Answer_End>", text, re.S):
            extract_ans = re.search(r"<Answer_Start>(.+?)<Answer_End>", text, re.S).group(1).strip()
        elif re.search(r"Answer:(.+?)$", text, re.S):
            extract_ans = re.search(r"Answer:(.+?)$", text, re.S).group(1).strip()
        elif re.search(r"答案:(.+?)$", text, re.S):
            extract_ans = re.search(r"答案:(.+?)$", text, re.S).group(1).strip()
        
        try:
            return strip_string(extract_ans)
        except:
            return extract_ans
        
    # @classmethod
    # def extract_answer(cls, text: str):
    #     extract_ans = super().extract_answer(text)
    #     if re.search(r'(?:-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?)', extract_ans):
    #         extract_ans = re.findall(r'(?:-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?)', extract_ans)[-1]
    #     return extract_ans

if __name__ == "__main__":
    data_params = BOOTSTRAPPING_PARAMS["gsm8k"]
    processor = BaseProcessor(data_params)
    processor.load_data("test")
    print("case:")
    print(processor.test_examples[0])
    examples = processor.prompting(processor.test_examples, "zeroshot_cot")
    print(examples[0])
