import argparse
import ujson
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--system_mode', type=str, choices=['baseline', 'system_mode'])
    args = parser.parse_args()
    return args

args = parse_args()
model_name = args.model_name


# deepseek-ai/deepseek-llm-7b-chat
# deepseek-ai/deepseek-llm-67b-chat
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

def get_prediction(messages: list):
    input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

    content = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    return content




if __name__ == '__main__':
    for filename in ['template_based', 'composition_based', 'llm_based']:
        target_dir = model_name.replace('/', '-')
        for round in ['round1', 'round2', 'round3']:
            start_time = time.time()
            print('begining:', start_time)
            with open(f'./prompts/{filename}.json', 'r', encoding='utf-8') as f:
                data = ujson.load(f)
            saved_dir = f'./out/{args.system_mode}/{target_dir}/{round}'
            os.makedirs(saved_dir, exist_ok=True)
            for item in data:
                prompt = item['prompt']
                messages = []
                if args.system_mode == 'system_mode':
                    messages.append({'role': 'system', 'content': 'You are a helpful assistant.'})
                messages.append({'role': 'user', "content": prompt})

                content = get_prediction(messages=messages)
                print(content)
                item['generation'] = content
                with open(f'./{saved_dir}/{filename}.json', 'w',
                        encoding='utf-8') as f:
                    ujson.dump(data, f, ensure_ascii=False, indent=2)
            end_time = time.time()
            print('ending:', end_time)
            print('total time:', filename, end_time - start_time)
