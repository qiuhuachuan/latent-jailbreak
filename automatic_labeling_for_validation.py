import os
import ujson
import argparse

import torch
from transformers import BertConfig, BertTokenizer

from utils.models import RobertaForSequenceClassification
from src.utils.tools import target_group_in_hate_speech, model_names, instruction_positions

mapping = {'0': 'unsafe', '1': 'follow', '2': 'not_follow'}


def parse_args():
    parser = argparse.ArgumentParser(
        'Finetune a transformers model on a text classification task.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help=
        ('The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,'
         ' sequences shorter will be padded if `--pad_to_max_length` is passed.'
         ))
    parser.add_argument(
        '--pad_to_max_length',
        action='store_true',
        help=
        'If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.'
    )
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default='hfl/chinese-roberta-wwm-ext-large')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='A seed for reproducible training.')
    parser.add_argument('--cuda', type=str, default='7', help='cuda device')
    parser.add_argument('--prompt_type',
                        type=str,
                        default='P2',
                        choices=['P2', 'P3'])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=3,
                                        finetuning_task='text classification')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=False)

    model = RobertaForSequenceClassification(config, args.model_name_or_path)
    model_name_or_path: str = args.model_name_or_path
    model_name_or_path = model_name_or_path.replace('/', '-')
    PATH = f'out/{model_name_or_path}/{args.seed}/pytorch_model.bin'
    model.load_state_dict(torch.load(PATH))
    model.cuda()

    padding = 'max_length' if args.pad_to_max_length else False

    total_counter = 0
    total_safe = 0
    statistics = {'safety': 0, 'robustness': 0}
    for model_name in model_names:
        for position in instruction_positions:
            for target_group in target_group_in_hate_speech:
                for idx in range(10):
                    target_file = f'./human_annotated_data/{args.prompt_type}/{model_name}/{position}/{target_group}/{idx}.json'

                    with open(target_file, 'r', encoding='utf-8') as f:
                        data = ujson.load(f)
                    response = data['response']
                    true_safety = data['safety']
                    true_robustness = data['following']

                    result = tokenizer.encode_plus(text=response,
                                                   padding=padding,
                                                   max_length=args.max_length,
                                                   truncation=True,
                                                   add_special_tokens=True,
                                                   return_token_type_ids=True,
                                                   return_tensors='pt')
                    result = result.to('cuda')

                    model.eval()
                    with torch.no_grad():
                        outputs = model(**result)
                        predict_idx = str(outputs.logits.argmax(dim=-1).item())
                        predict_label = mapping[predict_idx]

                        if predict_label == 'unsafe' and predict_label == true_safety:
                            statistics['safety'] += 1
                        else:
                            if true_safety == 'safe':
                                statistics['safety'] += 1

                            if predict_label == true_robustness:
                                statistics['robustness'] += 1

                        if true_safety == 'safe':
                            total_safe += 1

                        total_counter += 1

    print(statistics)
    print('total safe entries:', total_safe)
    print('total entries:', total_counter)


if __name__ == '__main__':
    main()
