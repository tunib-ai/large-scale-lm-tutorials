"""
src/process_group_3.py
"""

import torch.multiprocessing as mp
import torch.distributed as dist
import os


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, world_size):
    # rank는 기본적으로 들어옴. world_size는 입력됨.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # 프로세스 그룹 초기화
    group = dist.new_group([_ for _ in range(world_size)])
    # 프로세스 그룹 생성
    print(f"{group} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "4"

    mp.spawn(
        fn=fn,
        args=(4,),  # world_size 입력
        nprocs=4,  # 만들 프로세스 개수
        join=True,  # 프로세스 join 여부
        daemon=False,  # 데몬 여부
        start_method="spawn",  # 시작 방법 설정
    )
