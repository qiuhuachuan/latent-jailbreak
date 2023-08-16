import ujson

from datasets import Dataset, DatasetDict


def load_data(file_name: str):
    data_wrapper = {}

    instances = {'response': [], 'label': []}
    with open(f'./finetune_dataset/{file_name}.json', 'r',
              encoding='utf-8') as f:
        data = ujson.load(f)

    for item in data:
        response = item['response']
        label = item['label']
        instances['response'].append(response)
        instances['label'].append(label)
    data_wrapper[file_name] = Dataset.from_dict(instances)

    dataset = DatasetDict(data_wrapper)
    return dataset


if __name__ == '__main__':
    '''
    To test, run this file in the root folder.
    `python ./utils/dataloader.py`
    '''
    dataset = load_data(file_name='train')
    print(dataset)
