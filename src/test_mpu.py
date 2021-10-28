"""
src/test_mpu.py
"""
import torch
import torch.distributed as dist
from mpu import MPU

mpu = MPU(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    backend="nccl",
    master_port=5678,
)

global_rank = dist.get_rank()

# 1. MPU는 다음과 같이 프로세스 그룹을 자동으로 생성하고 그들에 접근할 수 있는 메서드를 제공합니다.
print(f"{global_rank}: TP group: {mpu.get_tensor_model_parallel_group()}")
print(f"{global_rank}: TP wsz: {mpu.get_tensor_model_parallel_world_size()}")
print(f"{global_rank}: TP rank: {mpu.get_tensor_model_parallel_rank()}")
dist.barrier()
print("\n")

print(f"{global_rank}: PP group: {mpu.get_pipeline_model_parallel_group()}")
print(f"{global_rank}: PP wsz: {mpu.get_pipeline_model_parallel_world_size()}")
print(f"{global_rank}: PP rank: {mpu.get_pipeline_model_parallel_rank()}")
dist.barrier()
print("\n")

# 2. Data parallel size는 TP와 PP 사이즈에 맞게 알아서 설정됩니다.
# 만약 16대의 GPU에서 TP=4, PP=1을 설정했다면 16 / (4 * 1) = 4로 자동으로 DP size는 4가 됩니다.
print(f"{global_rank}: DP group: {mpu.get_data_parallel_group()}")
print(f"{global_rank}: DP wsz: {mpu.get_data_parallel_world_size()}")
print(f"{global_rank}: DP rank: {mpu.get_data_parallel_rank()}")
dist.barrier()
print("\n")

# 3. MPU는 reduce, scatter, gather, braodcast 등의 연산을 지원합니다.
# 이들은 대부분 Tensor parallel group에서만 사용되기 때문에 Tensor parallel group을 기본으로 설장해뒀습니다.
tensor = torch.tensor([2, 3, 4, 5]).cuda() * global_rank
tensor = mpu.reduce(tensor)
print(f"{global_rank}: all-reduce => {tensor}")
