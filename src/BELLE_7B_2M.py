import os
import ujson
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils.tools import target_group_in_hate_speech, prompt_set

DEVICE = 'cuda'
DEVICE_ID = '6'
CUDA_DEVICE = f'{DEVICE}:{DEVICE_ID}' if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', default=None, type=str)
    parser.add_argument('--model', default='BELLE-7B-2M', type=str)
    args = parser.parse_args()

    return args


args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(f'BelleGroup/{args.model}',
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(f'BelleGroup/{args.model}',
                                             device_map=CUDA_DEVICE,
                                             trust_remote_code=True)
model.eval()


def data_store(data: dict, target_dir: str, idx: int) -> None:
    with open(f'{target_dir}/{idx}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)


def belle_7B_2M_generation(prompt: str, idx: int, target_dir: str,
                           context: str) -> None:
    inputs = tokenizer(prompt, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    inputs = inputs.to(CUDA_DEVICE)
    pred = model.generate(**inputs,
                          max_new_tokens=200,
                          do_sample=True,
                          top_k=30,
                          top_p=0.85,
                          temperature=0.35,
                          repetition_penalty=1.2)

    response = tokenizer.decode(pred.cpu()[0][input_length:],
                                skip_special_tokens=True)
    answer = {}
    answer['context'] = context
    answer['response'] = response

    torch_gc()
    data_store(data=answer, target_dir=target_dir, idx=idx)


if __name__ == '__main__':
    if args.prompt_type is None:
        prompt_types = prompt_set.keys()
    else:
        prompt_types = [args.prompt_type]

    for prompt_type in prompt_types:
        for position in prompt_set[prompt_type].keys():
            for target_group in target_group_in_hate_speech:

                target_dir = f'../raw_data/{prompt_type}/{args.model}/{position}/{target_group}'
                os.makedirs(target_dir, exist_ok=True)
                existing_files = os.listdir(target_dir)
                prompt: str = prompt_set[prompt_type][position]
                formatted_prompt = prompt.format(target_group)

                user_prompt = 'Human: ' + formatted_prompt + '\n\nAssistant:'

                for idx in range(30):
                    if f'{idx}.json' in existing_files:
                        print(f'DONE: {idx}.json')
                    else:
                        belle_7B_2M_generation(prompt=user_prompt,
                                               idx=idx,
                                               target_dir=target_dir,
                                               context=formatted_prompt)
                        print(f'SUCCESS: {target_dir}/{idx}.json')
