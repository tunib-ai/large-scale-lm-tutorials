"""
src/multi_process_1.py

참고:
Jupyter notebook은 멀티프로세싱 애플리케이션을 구동하는데에 많은 제약이 있습니다.
따라서 대부분의 경우 이곳에는 코드만 동봉하고 실행은 `src` 폴더에 있는 코드를 동작시키겠습니다.
실제 코드 동작은 `src` 폴더에 있는 코드를 실행시켜주세요.
"""

import torch.multiprocessing as mp
# 일반적으로 mp와 같은 이름을 사용합니다.


# 서브프로세스에서 동시에 실행되는 영역
def fn(rank, param1, param2):
    print(f"{param1} {param2} - rank: {rank}")


# 메인 프로세스
if __name__ == "__main__":
    processes = []
    # 시작 방법 설정
    mp.set_start_method("spawn")

    for rank in range(4):
        process = mp.Process(target=fn, args=(rank, "A0", "B1"))
        # 서브프로세스 생성
        process.daemon = False
        # 데몬 여부 (메인프로세스 종료시 함께 종료)
        process.start()
        # 서브프로세스 시작
        processes.append(process)

    for process in processes:
        process.join()
        # 서브 프로세스 join (=완료되면 종료)
