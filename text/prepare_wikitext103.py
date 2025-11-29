"""
WikiText-103 Raw 数据集准备脚本
使用 GPT-2 BPE tokenizer

用法:
  python prepare_wikitext103.py           # 完整数据 (集群)
  python prepare_wikitext103.py --local   # 小数据集 (本地测试)

数据保存到: data/wikitext103_raw/
"""
import argparse
import os
import numpy as np
import tiktoken
from datasets import load_dataset

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "wikitext103_raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true', help='本地测试模式，使用小数据集')
args = parser.parse_args()

# 配置
if args.local:
    NUM_TRAIN_SAMPLES = 10000   # 本地测试：1万文档
    NUM_VAL_SAMPLES = 1000      # 验证集：1千文档
    print("=== 本地测试模式 ===")
else:
    NUM_TRAIN_SAMPLES = 1000000  # 集群：100万文档
    NUM_VAL_SAMPLES = None       # 使用全部验证集
    print("=== 集群完整模式 ===")

# 下载 WikiText-103 Raw
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# 使用 GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # 添加 EOT token
    return {'ids': ids, 'len': len(ids)}

# 处理数据集
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing",
    num_proc=4,
)

# 合并所有 tokens 并保存
for split, dset in tokenized.items():
    # 限制样本数量
    if split == 'train' and NUM_TRAIN_SAMPLES and len(dset) > NUM_TRAIN_SAMPLES:
        dset = dset.select(range(NUM_TRAIN_SAMPLES))
        print(f"Limited {split} to {NUM_TRAIN_SAMPLES:,} samples")
    elif split == 'validation' and NUM_VAL_SAMPLES and len(dset) > NUM_VAL_SAMPLES:
        dset = dset.select(range(NUM_VAL_SAMPLES))
        print(f"Limited {split} to {NUM_VAL_SAMPLES:,} samples")

    arr_len = np.sum(dset['len'], dtype=np.uint64)
    # nanoGPT expects 'val.bin', not 'validation.bin'
    out_split = 'val' if split == 'validation' else split
    filename = os.path.join(OUTPUT_DIR, f'{out_split}.bin')
    arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))

    idx = 0
    for example in dset:
        arr[idx : idx + example['len']] = example['ids']
        idx += example['len']
    arr.flush()
    print(f"{out_split}: {arr_len:,} tokens saved to {filename}")
