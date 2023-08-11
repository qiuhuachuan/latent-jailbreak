import ujson
from rich import print

from src.utils.tools import model_names, instruction_positions

statistics = {}
for prompt_type in ['P14', 'P15', 'P16', 'P17', 'P18']:
    statistics[prompt_type] = {}
    for model in model_names:
        statistics[prompt_type][model] = {}
        for position in instruction_positions:
            statistics[prompt_type][model][position] = {}
            for idx in range(30):
                target_file = f'./human_annotated_data/version1/{prompt_type}/{model}/{position}/{idx}.json'

                with open(target_file, 'r', encoding='utf-8') as f:
                    data = ujson.load(f)

                following = data['following']

                if following in statistics[prompt_type][model][position]:
                    statistics[prompt_type][model][position][following] += 1
                else:
                    statistics[prompt_type][model][position][following] = 1

print(statistics)