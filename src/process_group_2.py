"""
src/process_group_2.py
"""

import torch.distributed as dist
import os


# 일반적으로 이 값들도 환경변수로 등록하고 사용합니다.
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# 통신에 필요한 주소를 설정합니다.
os.environ["MASTER_ADDR"] = "localhost"  # 통신할 주소 (보통 localhost를 씁니다.)
os.environ["MASTER_PORT"] = "29500"  # 통신할 포트 (임의의 값을 설정해도 좋습니다.)

dist.init_process_group(backend="nccl", rank=0, world_size=1)
# 프로세스 그룹 초기화

process_group = dist.new_group([0])
# 0번 프로세스가 속한 프로세스 그룹 생성

print(process_group)