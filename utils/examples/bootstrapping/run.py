"""
Copyright: anonymous
Time: 2024.12.04
Function: main execute entrance of reasoning bootstrapping.
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
from processors.base import BaseProcessor
from examples.config.data_params import BOOTSTRAPPING_PARAMS
from reason.bootstrapping import SAMPLERS
from utils.answer_extractor import extract_answer, extract_answer_remain_boxed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="", help="The data name")
    parser.add_argument("--data_kind", type=str, default="", help="The data kind (train / dev / test)")
    parser.add_argument("--save_path", type=str, default="./", help="The path of output")
    parser.add_argument("--model_name_or_path", type=str, default="", help="The path of model")
    parser.add_argument("--model_version", type=str, default="Meta-Llama-3-70B-Instruct", help="")
    parser.add_argument("--model_name", type=str, default="llama3", help="")
    parser.add_argument("--prompt_kind", type=str, default="zeroshot_cot", help="The kind of prompt template")
    parser.add_argument("--save_batch_size", type=int, default=1, help="Save results before every batch inference")
    parser.add_argument("--max_length", type=int, default=4096, help="The max length of generated text")
    parser.add_argument("--complete_steps", type=int, default=2, help="The number of stop token that needs to complete at one step")
    parser.add_argument("--duplicate_n", type=int, default=128, help="The number of sampled generated answer per query")
    parser.add_argument("--step_end_token", type=str, default="\n\n", help="The token the represent the end of one origin step, e.g., '\n\n'")
    parser.add_argument("--width_per_step", type=str, default="1", help="The number of sampled generated thought per step")
    parser.add_argument("--depth", type=int, default=1, help="The max number of searching step")
    parser.add_argument("--bootstrap_method", type=str, help="The kind of bootstrap method, e.g., vanilla, beam, mcts")
    parser.add_argument("--add_answer_format", action="store_true", help="Whether obtain the gt format before answering")
    parser.add_argument("--top_p", type=float, default=1.0, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--start_n", type=int, default=0, help="用于调试，只选择第start_n开始的样本")
    parser.add_argument("--cut_n", type=int, default=None, help="用于调试，只从start_n开始使用cut_n个样本")
    args = parser.parse_args()

    # load data
    print("load data ...")
    assert args.data_name in BOOTSTRAPPING_PARAMS.keys(), "You must select one of the data from {}".format(", ".join(list(BOOTSTRAPPING_PARAMS.keys())))
    assert BOOTSTRAPPING_PARAMS[args.data_name]["data_path"][args.data_kind] is not None, "The {} has no {}".format(args.data_name, args.data_kind)
    data_params = BOOTSTRAPPING_PARAMS[args.data_name]
    data_processor = BaseProcessor(data_params)
    data_processor.load_data(args.data_kind)
    examples = data_processor.get_data(args.data_kind)

    
    if args.add_answer_format is True and "answer_format" not in args.prompt_kind:
        print("You have set to add answer format for each query, so I recommend you to use answer format cot in answer sampling.")
    

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
    sampler = SAMPLERS[args.bootstrap_method](
        data_name=args.data_name,
        llm_agent=llm_agent,
        bootstrap_method=args.bootstrap_method,
    )


    # 划分批次
    batch_num = int((len(examples) - 1) / args.save_batch_size) + 1
    batch_examples = [examples[i * args.save_batch_size: (i + 1) * args.save_batch_size] for i in range(batch_num)]

    for ei, cur_examples in enumerate(batch_examples):
        print("====== batch {} [total {}] ======".format(ei + 1, batch_num))

        if args.add_answer_format:
            print("obtain the answer format of the ground truth ...")
            ground_truth_answers = [example["labels"][0] for example in cur_examples]
            answer_formats = llm_agent.generate_answer_format(ground_truth_answers)
            for example, answer_format in zip(cur_examples, answer_formats):
                answer_format = extract_answer_remain_boxed(answer_format)
                if answer_format is None:
                    answer_format = "\\boxed{<text>}"
                example["answer_format"] = answer_format
        
        # prompting
        print("prompting ...")
        cur_examples = data_processor.prompting(cur_examples, args.prompt_kind)

        print("sampling ...")
        sampled_examples = sampler.sampling(
            cur_examples, 
            duplicate_n=args.duplicate_n,
            width_per_step=args.width_per_step, 
            depth=args.depth,
            complete_steps=args.complete_steps,
            step_end_token=args.step_end_token,
        )

        print("post processing ...")
        sampled_examples = sampler.post_parse(sampled_results=sampled_examples)

        # save
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        
        with open(os.path.join(args.save_path, "{}_{}_{}_sampled_{}_{}_{}_from{}to{}.jsonl".format(args.data_name, args.data_kind, args.model_version, args.duplicate_n, args.bootstrap_method, args.prompt_kind, args.start_n, end_n)), "a", encoding="utf-8") as fw:
            for example in tqdm(sampled_examples):
                fw.write(json.dumps(example, ensure_ascii=False) + "\n")

