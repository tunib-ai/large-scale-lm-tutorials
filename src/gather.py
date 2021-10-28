"""
src/gather.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("gloo")
# nccl은 gather를 지원하지 않습니다.
rank = dist.get_rank()
torch.cuda.set_device(rank)

input = torch.ones(1) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

if rank == 0:
    outputs_list = [torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)]
    dist.gather(input, gather_list=outputs_list, dst=0)
    print(outputs_list)
else:
    dist.gather(input, dst=0)

