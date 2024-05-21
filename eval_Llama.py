import argparse
import ujson
import os
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--system_mode', type=str, choices=['baseline', 'system_mode'])
    args = parser.parse_args()
    return args

args = parse_args()
model_name = args.model_name

# Undi95/Meta-Llama-3-8B-Instruct-hf
# meta-llama/Meta-Llama-3-8B-Instruct
# meta-llama/Meta-Llama-3-70B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


def get_prediction(messages: list):
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=1000,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    content = tokenizer.decode(response, skip_special_tokens=True)
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
