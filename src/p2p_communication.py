"""
src/p2p_communication.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# 현재 nccl은 send, recv를 지원하지 않습니다. (2021/10/21)

if dist.get_rank() == 0:
    tensor = torch.randn(2, 2)
    dist.send(tensor, dst=1)

elif dist.get_rank() == 1:
    tensor = torch.zeros(2, 2)
    print(f"rank 1 before: {tensor}\n")
    dist.recv(tensor, src=0)
    print(f"rank 1 after: {tensor}\n")

else:
    raise RuntimeError("wrong rank")
