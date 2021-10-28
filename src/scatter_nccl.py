"""
src/scatter_nccl.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
inputs = torch.split(tensor=inputs, dim=-1, split_size_or_sections=1)
output = inputs[rank].contiguous().to(torch.cuda.current_device())
print(f"after rank {rank}: {output}\n")