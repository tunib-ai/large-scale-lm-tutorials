"""
src/zero_args.py
"""
from argparse import ArgumentParser
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import deepspeed
import torch.distributed as dist

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

parser = ArgumentParser()
parser.add_argument(
    "--deepspeed_config", default="../src/zero_dp_config.json", type=str
)
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
)

datasets = load_dataset("squad").data["train"]["context"]
datasets = [str(sample) for sample in datasets]
data_loader = DataLoader(datasets, batch_size=8, num_workers=8)

for i, data in enumerate(data_loader):
    tokens = tokenizer(
        data,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024,
    )

    loss = engine(
        input_ids=tokens.input_ids.cuda(),
        attention_mask=tokens.attention_mask.cuda(),
        labels=tokens.input_ids.cuda(),
    ).loss

    engine.backward(loss)
    engine.step()

    if i % 10 == 0 and dist.get_rank() == 0:
        print(f"step:{i}, loss:{loss}")

    if i >= 300:
        break
