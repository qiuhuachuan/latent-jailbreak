import os
import ujson
import random
import itertools

import numpy as np

from src.utils.tools import target_group_in_hate_speech, model_names, instruction_positions

random.seed(42)
np.random.seed(42)

train = []
statistics = {}
for model in model_names:
    for target_group in target_group_in_hate_speech:
        statistics[target_group] = {'safety': {}, 'robustness': {}}

        for position in instruction_positions:
            for idx in range(30):
                target_file = f'./human_annotated_data/version1/P1/{model}/{position}/{target_group}/{idx}.json'

                with open(target_file, 'r', encoding='utf-8') as f:
                    data = ujson.load(f)
                response = data['response']
                safety = data['safety']
                following = data['following']

                if safety == 'unsafe':
                    if safety in statistics[target_group]['safety']:
                        statistics[target_group]['safety'][safety].append(
                            response)
                    else:
                        statistics[target_group]['safety'][safety] = [response]

                else:
                    if following in statistics[target_group]['robustness']:
                        statistics[target_group]['robustness'][
                            following].append(response)
                    else:
                        statistics[target_group]['robustness'][following] = [
                            response
                        ]

        # print(statistics)
        for target_group in target_group_in_hate_speech:
            if target_group in statistics:
                # print(statistics[target_group]['safety'].keys())
                for safety_type in statistics[target_group]['safety'].keys():
                    samples = list(
                        set(statistics[target_group]['safety'][safety_type]))
                    train += list(itertools.combinations(samples, 2))

                for robustness_type in statistics[target_group][
                        'robustness'].keys():
                    samples = list(
                        set(statistics[target_group]['robustness']
                            [robustness_type]))
                    train += list(itertools.combinations(samples, 2))
with open('./finetune_dataset/train.jsonl', 'w', encoding='utf-8') as f:
    for el in train:
        sample = {'instruction': el[0], 'output': el[1]}
        f.write(ujson.dumps(sample, ensure_ascii=False) + '\n')