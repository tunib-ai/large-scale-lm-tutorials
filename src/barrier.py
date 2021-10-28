"""
src/barrier.py
"""
import time
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

if rank == 0:
    seconds = 0
    while seconds <= 3:
        time.sleep(1)
        seconds += 1
        print(f"rank 0 - seconds: {seconds}\n")

print(f"rank {rank}: no-barrier\n")
dist.barrier()
print(f"rank {rank}: barrier\n")