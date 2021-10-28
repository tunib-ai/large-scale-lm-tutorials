"""
src/efficient_data_parallel.py
"""

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset

# 1. create dataset
datasets = load_dataset("multi_nli").data["train"]
datasets = [
    {
        "premise": str(p),
        "hypothesis": str(h),
        "labels": l.as_py(),
    }
    for p, h, l in zip(datasets[2], datasets[5], datasets[9])
]
data_loader = DataLoader(datasets, batch_size=128, num_workers=4)

# 2. create model and tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3).cuda()

# 3. make data parallel module
# device_ids: 사용할 디바이스 리스트 / output_device: 출력값을 모을 디바이스
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3], output_device=0)

# 4. create optimizer
optimizer = Adam(model.parameters(), lr=3e-5)

# 5. start training
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

    loss = loss.mean()
    # (4,) -> (1,)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"step:{i}, loss:{loss}")

    if i == 300:
        break
