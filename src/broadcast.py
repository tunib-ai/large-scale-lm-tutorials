"""
src/broadcast.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
# device를 setting하면 이후에 rank에 맞는 디바이스에 접근 가능합니다.

if rank == 0:
    tensor = torch.randn(2, 2).to(torch.cuda.current_device())
else:
    tensor = torch.zeros(2, 2).to(torch.cuda.current_device())

print(f"before rank {rank}: {tensor}\n")
dist.broadcast(tensor, src=0)
print(f"after rank {rank}: {tensor}\n")