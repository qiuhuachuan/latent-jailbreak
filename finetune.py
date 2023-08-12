import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import pandas as pd
from uniem.finetuner import FineTuner
from accelerate.utils import set_seed

set_seed(42)

# 读取 jsonl 文件
df = pd.read_json('./finetune_dataset/train.jsonl', lines=True)
# 重新命名
df = df.rename(columns={'instruction': 'text', 'output': 'text_pos'})
# 指定训练的模型为 m3e-base
finetuner = FineTuner('moka-ai/m3e-base', dataset=df.to_dict('records'))
finetuner.run(epochs=3, output_dir='finetuned_model')