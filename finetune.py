import os
import argparse
import logging
import math
from tqdm.auto import tqdm

import torch
import evaluate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (SchedulerType, BertConfig, BertTokenizer,
                          default_data_collator, DataCollatorWithPadding,
                          get_scheduler)
from torch.utils.data import DataLoader

import wandb

from utils.dataloader import load_data
from utils.models import RobertaForSequenceClassification

logger = get_logger(__name__)


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
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.')
    parser.add_argument(
        '--per_device_valid_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the valid dataloader.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.'
    )
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.01,
                        help='Weight decay to use.')
    parser.add_argument('--num_train_epochs',
                        type=int,
                        default=10,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps',
                        type=int,
                        default=None,
                        help='Total number of training steps to perform.')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        'Number of updates steps to accumulate before performing a backward/update pass.'
    )
    parser.add_argument('--lr_scheduler_type',
                        type=SchedulerType,
                        default='linear',
                        help='The scheduler type to use.',
                        choices=[
                            'linear', 'cosine', 'cosine_with_restarts',
                            'polynomial', 'constant', 'constant_with_warmup'
                        ])
    parser.add_argument(
        '--num_warmup_steps',
        type=int,
        default=0,
        help='Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='A ratio for warmup steps.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='A seed for reproducible training.')
    parser.add_argument('--cuda', type=str, default='7', help='cuda device')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    wandb.init(project=args.model_name_or_path.replace('/', '-'))

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()

    datasets = load_data(file_name='train')

    label_list = (datasets['train']).unique('label')
    label_list.sort()
    num_labels = len(label_list)
    print(f'number of labels: {num_labels}')

    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task='text classification')

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=False)

    model = RobertaForSequenceClassification(config, args.model_name_or_path)

    padding = 'max_length' if args.pad_to_max_length else False

    def preprocess_function(examples):
        # tokenize the texts
        response = examples['response']
        result = tokenizer(text=response,
                           padding=padding,
                           max_length=args.max_length,
                           truncation=True,
                           add_special_tokens=True,
                           return_token_type_ids=True)
        result['labels'] = examples['label']
        return result

    with accelerator.main_process_first():
        process_datasets = datasets.map(preprocess_function,
                                        batched=True,
                                        remove_columns=['response', 'label'],
                                        desc='running tokenizer on dataset')

    train_dataset = process_datasets['train']

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=(8 if Accelerator.mixed_precision == 'fp16' else
                                None))

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  collate_fn=data_collator,
                                  batch_size=args.per_device_train_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            args.weight_decay,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                  lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {args.per_device_train_batch_size}'
    )
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}'
    )
    logger.info(
        f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(0, args.num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            wandb.log({'each epoch training loss': loss.item()})
        average_training_loss = total_loss / len(train_dataset)
        wandb.log({'average_training_loss': average_training_loss})

        if epoch == args.num_train_epochs - 1:

            model_name_or_path: str = args.model_name_or_path
            model_name_or_path = model_name_or_path.replace('/', '-')
            target_dir = f'./out/{model_name_or_path}/{args.seed}'
            os.makedirs(target_dir, exist_ok=True)

            target_file = f'{target_dir}/pytorch_model.bin'
            torch.save(model.state_dict(), target_file)


if __name__ == '__main__':
    main()
