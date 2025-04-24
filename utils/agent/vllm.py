"""
Copyright: anonymous
Time: 2024.11.29
Function: vLLM Agent for accelerating generation and reasoning.
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from typing import Optional, Callable, List, Tuple, Union
from utils.template import get_template_and_fix_tokenizer
from utils.answer_extractor import extract_answer



class vLLMAgent:
    def __init__(self, 
                model_name_or_path: str,
                trust_remote_code: bool = True,
                tensor_parallel_size: int = 1,
                model_name: str = "llama3",
                gpu_memory_utilization: float = 0.90,
                dtype: str = "bfloat16",
                **kwargs):

        self.model = LLM(
            model=model_name_or_path,
            tokenizer=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs = 64
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.template = get_template_and_fix_tokenizer(self.tokenizer, model_name)


    # generate via instruction
    def generate(self,
            prompts: Union[List[str], str],
            tokenizer: PreTrainedTokenizer = None,
            system: str = None,
            generation_config: Optional[GenerationConfig] = None,
            duplicate_n: int = 1,
            stop_tokens = None,
            **kwargs):
            
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        system = self.template.default_system if system is system else system
        
        print_flag = 0
        results = list()

        messages = []
        
        for prompt in prompts:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            messages.append(text)
        
        if stop_tokens is None:
            stop_tokens = ["Question:"]
            if "stop_tokens" in kwargs.keys():
                stop_tokens.extend(kwargs["stop_tokens"])

        sampling_params = SamplingParams(
            temperature=kwargs["temperature"] if "temperature" in kwargs.keys() else 0.8, 
            top_p=kwargs["top_p"] if "top_p" in kwargs.keys() else 1.0, 
            max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs.keys() else 4096, 
            # n=duplicate_n,
            # best_of=duplicate_n,
            stop=stop_tokens,
        )
        outputs = self.model.generate(messages, sampling_params=sampling_params)
        
        for output in outputs:

            prompt = output.prompt
            # generated_texts = [gen_output.text for gen_output in output.outputs]
            generated_texts = output.outputs[0].text

            # results.append({
            #     "input": prompt,
            #     "responses": generated_texts,
            # })
            results.append(generated_texts)
        
        return results
    
    # given a prompt and incomplete output, finish the rest.
    def complete(self,
            prompts: Union[List[str], str],
            incomplete_responses = Union[List[str], str],
            tokenizer: PreTrainedTokenizer = None,
            system: str = None,
            generation_config: Optional[GenerationConfig] = None,
            duplicate_n: int = 1,
            complete_steps: int = 1,
            stop_tokens = ["\n\n"],
            **kwargs):
        """
        complete_steps: complete step number, if set -1, complete the rest util meeting eot token
        """

        def check_whether_finish(generated_texts):
            if generated_texts.endswith(self.tokenizer.eos_token):
                # 如果在多步推理中某一步出现了eot_token，则下次不再添加该文本
                return True
            if extract_answer(generated_texts) is not None:
                # 如果生成的内容中已经包含了预测的结果
                return True
            return False


        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        system = self.template.default_system if system is system else system
        
        assert complete_steps >= -1 and complete_steps != 0
        is_complete_rest = False
        if complete_steps == -1:
            complete_steps = 1
            is_complete_rest = True
        
        
        # print_flag = 0
        results = [""] * len(prompts)
        finished_index = []

        for ei, text in enumerate(incomplete_responses):
            # # 如果输入的incomplete response以eot token结尾，说明其不需要再进行推理了
            # if text.endswith(self.tokenizer.eos_token):
            #     finished_index.append(ei)
            is_finished = check_whether_finish(text)
            if is_finished:
                finished_index.append(ei)

        

        for _ in range(complete_steps):

            messages = []
            incomplete_index_mapping = list() # is a list a[i], i means the i-th incomplete response, a[i] means the corresponding index
        
            for ei, (prompt, incomplete_response) in enumerate(zip(prompts, incomplete_responses)):

                if ei not in finished_index:
                    # only inference for incomplete prompt.
                    # 仅对未完成推理的response进行续写，从而避免对已经推理完成的response耗费时间续写
                    incomplete_index_mapping.append(ei)

                    message = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": incomplete_response},
                    ]
                    text = tokenizer.apply_chat_template(
                        message,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    # remove the special token after the incomplete response.
                    text = text[:text.index(incomplete_response) + len(incomplete_response)]
                    text += stop_tokens[0]
                    messages.append(text)
                
            if is_complete_rest is True:
                sampling_params = SamplingParams(
                    temperature=kwargs["temperature"] if "temperature" in kwargs.keys() else 0.8, 
                    top_p=kwargs["top_p"] if "top_p" in kwargs.keys() else 1.0, 
                    max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs.keys() else 4096, 
                )
            else:
                sampling_params = SamplingParams(
                    temperature=kwargs["temperature"] if "temperature" in kwargs.keys() else 0.8, 
                    top_p=kwargs["top_p"] if "top_p" in kwargs.keys() else 1.0, 
                    max_tokens=kwargs["max_tokens"] if "max_tokens" in kwargs.keys() else 4096, 
                    stop=stop_tokens,
                )

            outputs = self.model.generate(messages, sampling_params=sampling_params)
            
            for incomplete_index, output in enumerate(outputs):
                # outputs保留的是所有未完成推理的，因此这里也只会更新这些未推理完的，所以需要获得其在原始列表中的index
                
                ei = incomplete_index_mapping[incomplete_index] # obtain the origin index

                if ei in finished_index:
                    # 如果在多步推理中某一步出现了eot_token，则下次不再添加该文本
                    continue
                
                prompt = output.prompt
                generated_texts = output.outputs[0].text

                incomplete_responses[ei] += stop_tokens[0] + generated_texts

                results[ei] += generated_texts + stop_tokens[0]

                is_finished = check_whether_finish(generated_texts)
                if is_finished:
                    finished_index.append(ei)
                
        
        # 清除掉末尾的token
        for ei in range(len(results)):
            result = results[ei]
            if result.endswith(stop_tokens[0]):
                results[ei] = result[:len(result) - len(stop_tokens[0])]
        
        return results

    
    def generate_answer_format(self, 
        answers: Union[List[str], str],
        tokenizer: PreTrainedTokenizer = None,
        system: str = None,
        generation_config: Optional[GenerationConfig] = None
    ):
        # give an answer, obtain the format by rewriting it to an incorrect answer but share the very same format.
        prompt = """please rewrite the answer "{answer}" into an incorrect answer, please note that:\n1. the rewritten incorrect answer must has the very same format with the origin answer;\n2. The rewritten answer must be incorrect and has different tokens or words;\n3. Please directly output the incorrect answer in \\boxed."""
        prompts = [prompt.format(answer=answer) for answer in answers]
        answer_formats = self.generate(prompts, temperature=1.0, top_p=1.0)
        print("Show some cases: ")
        for i in range(min(5, len(answer_formats))):
            print("Origin answer (gt): {}\tGenerated format: {}".format(answers[i], answer_formats[i]))
        return answer_formats


    def chat(self, messages, tokenizer=None):
        tokenizer = self.tokenizer if tokenizer is None else tokenizer
        tokenized_messages = []
        inputs = []
        responses = []
        for message in messages:
            text = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            tokenized_messages.append(text)
        
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=4096, stop='<|eot_id|>')
        outputs = self.model.generate(tokenized_messages, sampling_params=sampling_params)
        for output in outputs:
            # print_flag = print_flag + 1
            prompt = output.prompt
            generated_text = output.outputs[0].text
            inputs.append(prompt)
            responses.append(generated_text)
        
        return inputs, responses

    def Self_ORM_Verfier(self):
        pass
    
    def Self_PRM_Verifier(self):
        pass