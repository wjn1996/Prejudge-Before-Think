import sys
import os
import torch
from reason.bootstrapping.base import BaseSampler


class IOSampler(BaseSampler):

    def description(self):
        print("This is the vanilla input-output generation.")

    def sampling(
        self, 
        examples,
        duplicate_n: int = 1,
        **kwargs):
        
        # obtain prompts
        examples_num = len(examples)

        prompts = [example["input"] for example in examples for _ in range(duplicate_n)]
        # labels = [example["labels"] for example in examples]

        # sampling generation
        example_results = self.llm_agent.generate(prompts=prompts)
        
        merged_example_results = list()
        for ei in range(examples_num):
            merged_example_results.append(example_results[ei * duplicate_n: (ei + 1) * duplicate_n])

        assert len(merged_example_results) == examples_num
        # merging
        sampled_examples = list()
        for example, cur_example_results in zip(examples, merged_example_results):
            sampled_examples.append({
                **example,
                "completions": cur_example_results,
                "bootstrap_method": "IO",
            })
        return sampled_examples

