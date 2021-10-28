"""
src/reduce_scatter.py
"""

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)

input_list = torch.tensor([1, 10, 100, 1000]).to(torch.cuda.current_device()) * rank
input_list = torch.split(input_list, dim=0, split_size_or_sections=1)
# rank==0 => [0, 00, 000, 0000]
# rank==1 => [1, 10, 100, 1000]
# rank==2 => [2, 20, 200, 2000]
# rank==3 => [3, 30, 300, 3000]

output = torch.tensor([0], device=torch.device(torch.cuda.current_device()),)

dist.reduce_scatter(
    output=output,
    input_list=list(input_list),
    op=torch.distributed.ReduceOp.SUM,
)

print(f"rank {rank}: {output}\n")
