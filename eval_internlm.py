import argparse
import ujson
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--system_mode', type=str, choices=['baseline', 'system_mode'])
    args = parser.parse_args()
    return args

args = parse_args()
model_name = args.model_name


# internlm/internlm2-chat-1_8b
# internlm/internlm2-chat-7b
# internlm/internlm2-chat-20b
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()


def get_prediction(text: str):
    response, history = model.chat(tokenizer, text, do_sample=True, history=[], max_new_tokens=1000)
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
                text = ''
                if args.system_mode == 'system_mode':
                    text += 'You are a helpful assistant.\n'
                text += prompt
                content = get_prediction(text=text)
                print(content)
                item['generation'] = content
                with open(f'./{saved_dir}/{filename}.json', 'w',
                        encoding='utf-8') as f:
                    ujson.dump(data, f, ensure_ascii=False, indent=2)
            end_time = time.time()
            print('ending:', end_time)
            print('total time:', filename, end_time - start_time)
