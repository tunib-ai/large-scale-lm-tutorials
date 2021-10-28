"""
src/pipe_dream.py
"""
import deepspeed
import torch
import torch.nn as nn
from datasets import load_dataset
from deepspeed import PipelineModule
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2BlockBase
import torch.distributed as dist


class GPT2Preprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)

    def forward(self, input_ids):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        position_ids = torch.arange(
            0, input_shape[-1], dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        return hidden_states


class GPT2Block(GPT2BlockBase):
    def forward(self, hidden_states):
        hidden_states = super(GPT2Block, self).forward(
            hidden_states=hidden_states,
        )
        return hidden_states[0]


class GPT2Postprocessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, hidden_states):
        hidden_states = self.ln_f(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        return lm_logits


def create_model_from_pretrained(model_name):
    pretrained = GPT2LMHeadModel.from_pretrained(model_name)
    preprocess = GPT2Preprocessing(pretrained.config)
    preprocess.wte.weight = pretrained.transformer.wte.weight
    preprocess.wpe.weight = pretrained.transformer.wpe.weight

    blocks = pretrained.transformer.h
    for i, block in enumerate(blocks):
        block.__class__ = GPT2Block

    postprocess = GPT2Postprocessing(pretrained.config)
    postprocess.ln_f.weight = pretrained.transformer.ln_f.weight
    postprocess.ln_f.bias = pretrained.transformer.ln_f.bias
    postprocess.lm_head.weight.data = pretrained.lm_head.weight.data.clone()

    return nn.Sequential(preprocess, *blocks, postprocess)


def collate_fn(batch):
    batch_encoding = tokenizer.pad(
        {"input_ids": batch}, padding="max_length", max_length=1024
    )
    return batch_encoding.input_ids


def batch_fn(data):
    input_ids = data
    labels = data
    return input_ids, labels


def loss_fn(logits, labels):
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    return nn.CrossEntropyLoss()(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size, rank = dist.get_world_size(), dist.get_rank()
    batch_size, train_steps = 16, 300
    train_samples = batch_size * train_steps

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = PipelineModule(
        create_model_from_pretrained(model_name="gpt2"),
        loss_fn=loss_fn,
        num_stages=world_size,
        partition_method="type:GPT2Block"
        # partition_method를 통해 병렬화 하고 싶은 레이어를 고를 수 있습니다.
    )
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=Adam(model.parameters(), lr=3e-5),
        config={
            "train_batch_size": batch_size,
            "steps_per_print": 9999999,
            # turn off: https://github.com/microsoft/DeepSpeed/issues/1119
        },
    )
    engine.set_batch_fn(batch_fn)

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for i, sample in enumerate(datasets) if i < train_samples]
    datasets = [
        tokenizer(data, return_tensors="pt", max_length=1024).input_ids[0]
        for data in tqdm(datasets)
    ]
    data_loader = iter(
        DataLoader(
            sorted(datasets, key=len, reverse=True),
            # uniform length batching
            # https://mccormickml.com/2020/07/29/smart-batching-tutorial/
            batch_size=batch_size,
            num_workers=8,
            collate_fn=collate_fn,
            shuffle=False,
        )
    )

    for i in range(train_steps):
        loss = engine.train_batch(data_loader)
        # evaluation은 engine.eval_batch(data_loader)로 가능합니다.

        if i % 10 == 0 and rank == 0:
            print(f"step: {i}, loss: {loss}")
