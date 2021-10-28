"""
src/zero_args.py
"""
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import deepspeed
import torch.distributed as dist

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)

engine, optimizer, _, scheduler = deepspeed.initialize(
    optimizer=optimizer,
    model=model,
    config={
        "train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": 300,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 30,
            },
        },
        "fp16": {
            "enabled": True,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False,
        "steps_per_print": 9999999999,
    },
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
