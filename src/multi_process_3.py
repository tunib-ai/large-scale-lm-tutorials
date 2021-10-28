"""
src/multi_process_3.py
"""

# 코드 전체가 서브프로세스가 됩니다.
import os

# RANK, LOCAL_RANK, WORLD_SIZE 등의 변수가 자동으로 설정됩니다.
print(f"hello world, {os.environ['RANK']}")