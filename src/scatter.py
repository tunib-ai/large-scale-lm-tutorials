"""
src/scatter.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# nccl은 scatter를 지원하지 않습니다.
rank = dist.get_rank()
torch.cuda.set_device(rank)


output = torch.zeros(1)
print(f"before rank {rank}: {output}\n")

if rank == 0:
    inputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
    inputs = torch.split(inputs, dim=0, split_size_or_sections=1)
    # (tensor([10]), tensor([20]), tensor([30]), tensor([40]))
    dist.scatter(output, scatter_list=list(inputs), src=0)
else:
    dist.scatter(output, src=0)

print(f"after rank {rank}: {output}\n")
