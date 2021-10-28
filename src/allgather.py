"""
src/allgather.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input = torch.ones(1).to(torch.cuda.current_device()) * rank
# rank==0 => [0]
# rank==1 => [1]
# rank==2 => [2]
# rank==3 => [3]

outputs_list = [
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
    torch.zeros(1, device=torch.device(torch.cuda.current_device())),
]

dist.all_gather(tensor_list=outputs_list, tensor=input)
print(outputs_list)
