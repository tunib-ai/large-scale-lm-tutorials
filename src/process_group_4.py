"""
src/process_group_4.py
"""

import torch.distributed as dist

dist.init_process_group(backend="nccl")
# 프로세스 그룹 초기화

group = dist.new_group([_ for _ in range(dist.get_world_size())])
# 프로세스 그룹 생성

print(f"{group} - rank: {dist.get_rank()}\n")