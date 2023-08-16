import ujson

from src.utils.tools import target_group_in_hate_speech, model_names, instruction_positions

train = []

for model in model_names:
    for target_group in target_group_in_hate_speech:
        for position in instruction_positions:
            for idx in range(30):
                target_file = f'./human_annotated_data/version1/P1/{model}/{position}/{target_group}/{idx}.json'

                with open(target_file, 'r', encoding='utf-8') as f:
                    data = ujson.load(f)
                response = data['response']
                safety = data['safety']
                following = data['following']

                if safety == 'unsafe':
                    item = {'response': response, 'label': 0}
                else:
                    item = {
                        'response': response,
                        'label': 1 if following == 'follow' else 2
                    }
                train.append(item)

with open('./finetune_dataset/train.json', 'w', encoding='utf-8') as f:
    ujson.dump(train, f, ensure_ascii=False, indent=2)

print('done')