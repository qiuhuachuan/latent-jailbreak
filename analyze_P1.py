import ujson
from rich import print

from src.utils.tools import target_group_in_hate_speech, model_names, instruction_positions

statistics = {}
for model in model_names:
    statistics[model] = {'safety': {}, 'following': {}}
    for position in instruction_positions:
        for target_group in target_group_in_hate_speech:
            for idx in range(30):
                target_file = f'./human_annotated_data/P1/{model}/{position}/{target_group}/{idx}.json'

                with open(target_file, 'r', encoding='utf-8') as f:
                    data = ujson.load(f)
                safety = data['safety']
                following = data['following']

                if safety in statistics[model]['safety']:
                    statistics[model]['safety'][safety] += 1
                else:
                    statistics[model]['safety'][safety] = 1

                if following in statistics[model]['following']:
                    statistics[model]['following'][following] += 1
                else:
                    statistics[model]['following'][following] = 1

print(statistics)
