"""
src/megatron_datasets.py
"""

import json
import os
from datasets import load_dataset

train_samples, min_length = 10000, 512
filename = "megatron_datasets.jsonl"
curr_num_datasets = 0

if os.path.exists(filename):
    os.remove(filename)

datasets = load_dataset("wikitext", "wikitext-103-raw-v1")
datasets = datasets.data["train"]["text"]
dataset_fp_write = open(filename, mode="w", encoding="utf-8")

for sample in datasets:
    sample = sample.as_py()

    if len(sample) >= min_length:
        line = json.dumps(
            {"text": sample},
            ensure_ascii=False,
        )

        dataset_fp_write.write(line + "\n")
        curr_num_datasets += 1

        # 튜토리얼이기 때문에 적은 양의 데이터만 만들겠습니다.
        if curr_num_datasets >= train_samples:
            break

dataset_fp_read = open(filename, mode="r", encoding="utf-8")
dataset_read = dataset_fp_read.read().splitlines()[:3]

# 데이터의 구조를 확인합니다.
for sample in dataset_read:
    print(sample, end="\n\n")
