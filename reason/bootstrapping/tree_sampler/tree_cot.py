"""
Copyright: anonymous
Time: 2024.12.11
Function: bootstrapping with tree
"""

import sys
import os
import torch
from typing import Union, List
from reason.bootstrapping.base import BaseSampler
from reason.bootstrapping.tree_sampler.tree import CoTTree



class TreeSampler(BaseSampler):

    def description(self):
        print("This is the tree sampler. At each time, the agent needs to sampling K different steps for each thought path in the next time.")

    def sampling(
        self,
        examples,
        duplicate_n: int = 1,
        width_per_step: Union[str, List[int]] = "",
        depth: int = 1,
        step_end_token: str = "\n\n",
        complete_steps: int = 2,
        **kwargs):

        # width = duplicate_n
        depth = max(1, depth)
        """
        examples: List[Dict], must consists of "input"
        duplicate_n: the sampling branch number of each step.
        width_per_step: the sampling branch number of corresponding step. for example "2,3,4" means the 1st / 2nd / 3rd layer (step) need to sample 2 / 3 / 4 times.
        depth: the max number of step: When at the last steps, the model directly output the rest tokens util end.
        step_end_token: the token marking the end of each step. default as "\n\n".
        complete_steps: split the step when seeing k step_end_tokens, for example if complete_steps set 3, the "xxx\n\nxxx\n\nxxx\n\n" will be a step.
        """

        if type(width_per_step) == str:
            width_per_step = width_per_step.strip().replace(" ", "").split(",")
        width_per_step = [int(i) for i in width_per_step]

        if len(width_per_step) < depth:
            width_per_step += [duplicate_n] * (depth - len(width_per_step))
        
        assert len(width_per_step) == depth
        

        # obtain prompts
        examples_num = len(examples)
        prompts = [example["input"] for example in examples]

        sampled_results = list()
        sampled_trees = list()

        # 遍历每一个example
        for prompt in prompts:
            
            tree = CoTTree(prompt=prompt)

            
            # A. at the first, directly use generate function to generate response, and the agent may stop when facing step_end_token.
            layer_index = 0
            incomplete_responses = self.llm_agent.generate(
                prompts=[prompt] * width_per_step[layer_index], 
                stop_tokens=[step_end_token],
            )
            for ei, incomplete_response in enumerate(incomplete_responses):
                tree.set_node("{}".format(ei + 1), incomplete_response)
            
            layer_index += 1
            
            # B. at next time, use complete function to iteratively complete the generated response step by step.
            while layer_index < depth:
                
                # 1. obtain all leaf nodes and the corresponding imcomplete responses
                all_leaf_nodes = tree.get_all_leafs()
                all_cots = [tree.get_context(node)["response"] for node in all_leaf_nodes]                

                # 2. check which cot is incomplete (not ended with eos token, e.g. <|eot_id|> in llama3)
                incomplete_nodes, incomplete_cots = list(), list()
                for node, cot in zip(all_leaf_nodes, all_cots):
                    if not cot.endswith(self.tokenizer.eos_token):
                        incomplete_nodes.append(node)
                        incomplete_cots.append(cot)
                incomplete_nodes_num = len(incomplete_nodes)
                # 3. use complete function to complete the cot
                incomplete_nodes = incomplete_nodes * width_per_step[layer_index]
                incomplete_cots = incomplete_cots * width_per_step[layer_index]

                # if at the last iteration (depth-level) some cots are not finished after depth, complete the rest.
                if layer_index == depth - 1:
                    incomplete_responses = self.llm_agent.complete(
                        prompts=[prompt] * incomplete_nodes_num * width_per_step[layer_index],
                        incomplete_responses=incomplete_cots,
                        stop_tokens=[step_end_token],
                        complete_steps=-1,
                    )
                else:
                    incomplete_responses = self.llm_agent.complete(
                        prompts=[prompt] * incomplete_nodes_num * width_per_step[layer_index],
                        incomplete_responses=incomplete_cots,
                        stop_tokens=[step_end_token],
                        complete_steps=complete_steps,
                    )

                # 4. add new nodes in tree
                new_node_index = 1
                for i in range(0, len(incomplete_nodes), incomplete_nodes_num):
                    cur_incomplete_nodes = incomplete_nodes[i: i + incomplete_nodes_num]
                    cur_incomplete_responses = incomplete_responses[i: i + incomplete_nodes_num]
                    for incomplete_node, incomplete_response in zip(cur_incomplete_nodes, cur_incomplete_responses):
                        new_node = "{}-{}".format(incomplete_node, new_node_index)
                        tree.set_node(new_node, incomplete_response)
                    new_node_index += 1
                
                layer_index += 1
            
            # C. obtain all responses for the current prompt

            all_leaf_nodes = tree.get_all_leafs()
            all_cots = [tree.get_context(node)["response"] for node in all_leaf_nodes]
            sampled_results.append(all_cots)
            sampled_trees.append(tree.cot_tree)
        
        # merging
        sampled_examples = list()
        for example, cur_example_results, cur_example_tree in zip(examples, sampled_results, sampled_trees):
            sampled_examples.append({
                **example,
                "completions": cur_example_results,
                "bootstrap_method": self.bootstrap_method,
                "tree": cur_example_tree,
            })

        return sampled_examples

    def post_parse(self, sampled_results):
        """
        process and parse the sampled tree:
        1. obtain all leaf nodes and corresponding cot
        2. model eval to detect whether the answer is correct, and calculate the reward (value) of each node 
        """

        VERIFIER_TEMPLATE = """Given a problem and the corresponding ground truths, the task is to verify if the generated answer can match one of the candidate ground truths. Please output "TRUE" or "FALSE" only.
------
Below is the one you need to verify:
### Start of Problem
{PROBLEM}
### End of Problem
### Start of Generated Answer
{FINAL_ANSWER}
### End of Generated Answer
### Start of Ground Truth
{GROUND_TRUTH}
### End of Ground Truth
### Start of Verification"""
        
        for example in sampled_results:

            # 1. parsing tree

            tree = CoTTree(prompt=example["question"], cot_tree=example["tree"])
            tree.remove_blank_node()
            leaf_node_list = tree.get_all_leafs() # 获取所有叶子节点
            cots = [tree.get_context(node) for node in leaf_node_list] # 获取所有推理路径（根节点到叶子节点）
            # cots = [cot if cot is not None else "I can't answer." for cot in cots]


            # 2. model eval

            # verified_prompt = VERIFIER_TEMPLATE.format(
            #     PROBLEM=example["question"],
            #     GROUND_TRUTH="\n".join([f"Candidate {ei + 1}: {label}" for ei, label in enumerate(example["labels"])]),
            # )

            verified_prompt = VERIFIER_TEMPLATE.replace(
                "{PROBLEM}", 
                example["question"]
            ).replace(
                "{GROUND_TRUTH}", 
                "\n".join([f"Candidate {ei + 1}: {label}" for ei, label in enumerate(example["labels"])])
            )

            verified_prompt = [verified_prompt.replace("{FINAL_ANSWER}", cot["response"]) for cot in cots]

            verified_results = self.llm_agent.generate(prompts=verified_prompt)
            rewards = [1 if "TRUE" in verified_result else 0 for verified_result in verified_results]
            try:
                tree.set_reward(leaf_node_list, rewards, "max") # 将整个tree的每个内部节点和叶子节点都分配reward
            except:
                pass

            example["tree"] = tree.cot_tree
        
        return sampled_results

