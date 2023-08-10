import os
import ujson
import argparse
from typing import List

from transformers import AutoTokenizer, AutoModel
import torch

from utils.tools import target_group_in_hate_speech, prompt_set

DEVICE = "cuda"
DEVICE_ID = '5'
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', default=None, type=str)
    parser.add_argument('--model', default='chatglm2-6b', type=str)
    args = parser.parse_args()

    return args


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(f"THUDM/{args.model}",
                                          trust_remote_code=True)
model = AutoModel.from_pretrained(f"THUDM/{args.model}",
                                  device_map=CUDA_DEVICE,
                                  trust_remote_code=True)
model.eval()


def data_store(completion: dict, target_dir: str, idx: int) -> None:
    with open(f'{target_dir}/{idx}.json', 'w', encoding='utf-8') as f:
        ujson.dump(completion, f, ensure_ascii=False, indent=2)


def chatglm2_6B_generation(prompt: str,
                           idx: int,
                           history: List,
                           target_dir: str,
                           max_tokens: int = 2048) -> None:

    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_tokens,
                                   top_p=0.7,
                                   temperature=0.9)

    answer = {'context': prompt, 'response': response}

    torch_gc()
    data_store(completion=answer, target_dir=target_dir, idx=idx)


if __name__ == '__main__':
    if args.prompt_type is None:
        prompt_types = prompt_set.keys()
    else:
        prompt_types = [args.prompt_type]

    for prompt_type in prompt_types:
        for position in prompt_set[prompt_type].keys():
            for target_group in target_group_in_hate_speech:

                target_dir = f'../raw_data/{prompt_type}/ChatGLM2-6B/{position}/{target_group}'
                os.makedirs(target_dir, exist_ok=True)
                existing_files = os.listdir(target_dir)
                prompt: str = prompt_set[prompt_type][position]
                formatted_prompt = prompt.format(target_group)

                for idx in range(30):
                    if f'{idx}.json' in existing_files:
                        print(f'DONE: {idx}.json')
                    else:
                        chatglm2_6B_generation(prompt=formatted_prompt,
                                               idx=idx,
                                               history=[],
                                               target_dir=target_dir)
                        print(f'SUCCESS: {target_dir}/{idx}.json')
