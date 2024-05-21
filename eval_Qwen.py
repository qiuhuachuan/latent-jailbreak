import argparse
import ujson
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--system_mode', type=str, choices=['baseline', 'system_mode'])
    args = parser.parse_args()
    return args

args = parse_args()
model_name = args.model_name


# Qwen/Qwen1.5-1.8B-Chat
# Qwen/Qwen1.5-4B-Chat
# Qwen/Qwen1.5-7B-Chat
# Qwen/Qwen1.5-14B-Chat
# Qwen/Qwen1.5-32B-Chat
# Qwen/Qwen1.5-72B-Chat
# Qwen/Qwen1.5-110B-Chat
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


def get_prediction(messages: list):
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_ids = tokenizer(input_text, return_tensors="pt").to(f"cuda")
    start = len(tokenizer(input_text)['input_ids'])

    outputs = model.generate(**input_ids, max_new_tokens=1000)
    content = tokenizer.decode(outputs[0][start:-1])
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
