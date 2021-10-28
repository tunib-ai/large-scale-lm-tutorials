"""
src/process_group_1.py
"""

import torch.distributed as dist
# 일반적으로 dist와 같은 이름을 사용합니다.

dist.init_process_group(backend="nccl", rank=0, world_size=1)
# 프로세스 그룹 초기화
# 본 예제에서는 가장 자주 사용하는 nccl을 기반으로 진행하겠습니다.
# backend에 'nccl' 대신 'mpi'나 'gloo'를 넣어도 됩니다.

process_group = dist.new_group([0])
# 0번 프로세스가 속한 프로세스 그룹 생성

print(process_group)