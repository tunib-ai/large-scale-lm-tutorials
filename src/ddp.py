"""
src/ddp.py
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# 1. initialize process group
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

# 2. create dataset
datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]

# 3. create DistributedSampler
# DistributedSampler는 데이터를 쪼개서 다른 프로세스로 전송하기 위한 모듈입니다.
sampler = DistributedSampler(
    datasets,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)
data_loader = DataLoader(
    datasets,
    batch_size=32,
    num_workers=4,
    sampler=sampler,
    shuffle=False,
    pin_memory=True,
)


# 4. create model and tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()
# 5. make distributed data parallel module
model = DistributedDataParallel(model, device_ids=[device], output_device=device)

# 5. create optimizer
optimizer = Adam(model.parameters(), lr=3e-5)

# 6. start training
for i, data in enumerate(data_loader):
    optimizer.zero_grad()
    tokens = tokenizer(
        data["premise"],
        data["hypothesis"],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    loss = model(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=data["labels"],
    ).loss

    loss.backward()
    optimizer.step()

    if i % 10 == 0 and rank == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break
