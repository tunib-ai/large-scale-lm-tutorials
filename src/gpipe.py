"""
src/gpipe.py
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchgpipe import GPipe
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block as GPT2BlockBase


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


if __name__ == "__main__":
    world_size = 4

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = create_model_from_pretrained(model_name="gpt2")
    model = GPipe(
        model,
        balance=[4, 3, 3, 4],
        devices=[0, 1, 2, 3],
        chunks=world_size,
    )

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(sample) for sample in datasets]
    data_loader = DataLoader(datasets, batch_size=8, num_workers=8)

    optimizer = Adam(model.parameters(), lr=3e-5)
    loss_fn = nn.CrossEntropyLoss()

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        tokens = tokenizer(data, return_tensors="pt", truncation=True, padding=True)
        input_ids = tokens.input_ids.to(0)
        labels = tokens.input_ids.to(world_size - 1)

        lm_logits = model(input_ids)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss}")
        if i == 300:
            break
