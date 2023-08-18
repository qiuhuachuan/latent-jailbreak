import ujson
from rich import print

from src.utils.tools import target_group_in_hate_speech, model_names, instruction_positions, prompt_set

for prompt_type in prompt_set.keys():
    print(f'prompt type: {prompt_type}')
    statistics = {}
    for model in model_names:
        statistics[model] = {'safety': {}, 'following': {}}
        for position in instruction_positions:
            with open(
                    f'./data/{prompt_type}/{model}/{prompt_type}_{position}.json',
                    'r',
                    encoding='utf-8') as f:
                all_data = ujson.load(f)
            for target_group in target_group_in_hate_speech:
                for item in all_data[target_group]:
                    safety = item['safety']
                    following = item['following']

                    if safety in statistics[model]['safety']:
                        statistics[model]['safety'][safety] += 1
                    else:
                        statistics[model]['safety'][safety] = 1

                    if following in statistics[model]['following']:
                        statistics[model]['following'][following] += 1
                    else:
                        statistics[model]['following'][following] = 1
    print(statistics)
    print()