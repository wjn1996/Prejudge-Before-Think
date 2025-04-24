import sys
import os

class BaseSampler():
    def __init__(self, data_name, llm_agent, bootstrap_method=None):
        self.data_name = data_name
        self.llm_agent = llm_agent
        self.tokenizer = llm_agent.tokenizer
        self.bootstrap_method = None
    
    def description(self):
        print("Please write the description for this bootstrapping method.")

    def sampling(self, examples):
        """
        Use llm agent to sample generate cots
        """
        pass
    
    def post_parse(self, sampled_results):
        """
        Process the sampled results
        """
        return sampled_results