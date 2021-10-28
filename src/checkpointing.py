"""
src/checkpointing.py
"""
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import BertTokenizer, BertLayer, BertConfig

config = BertConfig.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer("Hello I am Kevin", return_tensors="pt")

embedding = nn.Embedding(tokenizer.vocab_size, config.hidden_size)
layers = nn.ModuleList([BertLayer(config) for _ in range(6)])

hidden_states = embedding(tokens.input_ids)
attention_mask = tokens.attention_mask

for i, layer_module in enumerate(layers):
    layer_outputs = checkpoint(
        layer_module,
        hidden_states,
        attention_mask,
    )

    hidden_states = layer_outputs[0]

print(f"output: {hidden_states}")