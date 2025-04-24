"""
Copyright: anonymous
Time: 2024.12.27
Function: main execute entrance of reasoning distillation.
"""

import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
import os
import jsonlines
import argparse
import torch
import json
from tqdm import tqdm
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Callable, List, Tuple, Union

from agent.vllm import vLLMAgent
from reason.bootstrapping import SAMPLERS
from processors.distillation import PrejudgeCritiqueProcessor
from examples.config.data_params import DISTILLATING_PARAMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="", help="The data name")
    parser.add_argument("--data_kind", type=str, default="", help="The data kind (train / dev / test)")
    parser.add_argument("--save_path", type=str, default="./", help="The path of output")
    parser.add_argument("--model_name_or_path", type=str, default="", help="The path of model")
    parser.add_argument("--model_version", type=str, default="Meta-Llama-3-70B-Instruct", help="")
    parser.add_argument("--model_name", type=str, default="llama3", help="")
    parser.add_argument("--save_batch_size", type=int, default=1, help="Save results before every batch inference")
    parser.add_argument("--max_length", type=int, default=4096, help="The max length of generated text")
    parser.add_argument("--duplicate_n", type=int, default=128, help="The number of sampled generated answer per query")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--start_n", type=int, default=0, help="用于调试，只选择第start_n开始的样本")
    parser.add_argument("--cut_n", type=int, default=None, help="用于调试，只从start_n开始使用cut_n个样本")
    args = parser.parse_args()

    # load data
    print("load data ...")
    assert args.data_name in DISTILLATING_PARAMS.keys(), "You must select one of the data from {}".format(", ".join(list(BOOTSTRAPPING_PARAMS.keys())))
    assert DISTILLATING_PARAMS[args.data_name]["data_path"][args.data_kind] is not None, "The {} has no {}".format(args.data_name, args.data_kind)
    data_params = DISTILLATING_PARAMS[args.data_name]
    data_processor = PrejudgeCritiqueProcessor(data_params)
    data_processor.load_data(args.data_kind)
    examples = data_processor.get_data(args.data_kind)

    # data cut
    print("data cutting ...")
    print("total_num=", len(examples))
    print("start_n=", args.start_n)
    print("cut_n=", args.cut_n)
    examples = examples[args.start_n:]
    if args.cut_n is not None or args.cut_n != -1:
        examples = examples[:args.cut_n]
    
    end_n = args.start_n + len(examples)
    print("end_n=", end_n)

    # load model
    print("load model")
    llm_agent = vLLMAgent(
        model_name_or_path=args.model_name_or_path,
        model_name=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
    )

    # load sampler
    sampler = SAMPLERS["io"]( # DO NOT CHANGE
        data_name=args.data_name,
        llm_agent=llm_agent,
        bootstrap_method="io",
    )

    # 划分批次
    batch_num = int((len(examples) - 1) / args.save_batch_size) + 1
    batch_examples = [examples[i * args.save_batch_size: (i + 1) * args.save_batch_size] for i in range(batch_num)]

    for ei, cur_examples in enumerate(batch_examples):
        print("====== batch {} [total {}] ======".format(ei + 1, batch_num))
        
        # prompting
        print("prompting ...")
        cur_examples = data_processor.prompting(cur_examples)

        print("distillating ...")
        sampled_examples = sampler.sampling(
            cur_examples, 
            duplicate_n=args.duplicate_n,
        )

        # save
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        with open(os.path.join(args.save_path, "{}_{}_{}_sampled_{}_{}_from{}to{}.jsonl".format(args.data_name, args.data_kind, args.model_version, args.duplicate_n, "distillating", args.start_n, end_n)), "a", encoding="utf-8") as fw:
            for example in tqdm(sampled_examples):
                fw.write(json.dumps(example, ensure_ascii=False) + "\n")