"""
src/multi_process_2.py
"""

import torch.multiprocessing as mp


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, param1, param2):
    # rank는 기본적으로 들어옴. param1, param2는 spawn시에 입력됨.
    print(f"{param1} {param2} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    mp.spawn(
        fn=fn,
        args=("A0", "B1"),
        nprocs=4,  # 만들 프로세스 개수
        join=True,  # 프로세스 join 여부
        daemon=False,  # 데몬 여부
        start_method="spawn",  # 시작 방법 설정
    )
