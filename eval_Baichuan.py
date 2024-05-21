import argparse
import ujson
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=['baichuan-inc/Baichuan2-7B-Chat', 'baichuan-inc/Baichuan2-13B-Chat'])
    parser.add_argument('--system_mode', type=str, choices=['baseline', 'system_mode'])
    args = parser.parse_args()
    return args

args = parse_args()
model_name = args.model_name


# baichuan-inc/Baichuan2-7B-Chat
# baichuan-inc/Baichuan2-13B-Chat
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.max_new_tokens = 1000

def get_prediction(messages: list):
    response = model.chat(tokenizer, messages)
    return response


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
