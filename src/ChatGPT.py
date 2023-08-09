import os
import time
import ujson
import argparse

import openai

from utils.tools import target_group_in_hate_speech, prompt_set


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--model', default='ChatGPT', type=str)
    args = parser.parse_args()

    return args


def data_store(data: dict, target_dir: str, idx: int) -> None:
    with open(f'{target_dir}/{idx}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)


def ChatGPT_generation(target_dir: str, idx: int, prompt: str,
                       max_tokens: int) -> None:
    messages = []
    user_msg = {'role': 'user', 'content': f'''{prompt}'''}
    messages.append(user_msg)
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613",
                                              temperature=1.0,
                                              top_p=1.0,
                                              messages=messages,
                                              max_tokens=max_tokens)
    answer = {}
    answer['context'] = prompt
    answer['response'] = (
        completion['choices'][0]['message']['content']).strip()

    data_store(data=answer, target_dir=target_dir, idx=idx)
    time.sleep(3)


if __name__ == '__main__':
    args = parse_args()
    openai.api_key = args.api_key
    for prompt_type in prompt_set.keys():
        for position in prompt_set[prompt_type].keys():
            for target_group in target_group_in_hate_speech:
                target_dir = f'../raw_data/{prompt_type}/{args.model}/{position}/{target_group}'
                os.makedirs(target_dir, exist_ok=True)
                existing_files = os.listdir(target_dir)
                prompt: str = prompt_set[prompt_type][position]
                formatted_prompt = prompt.format(target_group)
                user_msg = formatted_prompt

                for idx in range(30):
                    if f'{idx}.json' in existing_files:
                        print(f'DONE: {idx}.json')
                    else:
                        try:
                            ChatGPT_generation(target_dir=target_dir,
                                               idx=idx,
                                               prompt=prompt,
                                               max_tokens=200)
                            print(f'SUCCESS: {target_dir}/{idx}.json')
                        except Exception as e:
                            print(f'ERROR-INFO: {e}')
                            print(f'ERROR: {target_dir}/{idx}.json')
                            time.sleep(120)
