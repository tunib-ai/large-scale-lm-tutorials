"""
src/non_parallelism.py
"""

import torch

X = torch.tensor(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
)

A = torch.tensor(
    [
        [10, 14],
        [11, 15],
        [12, 16],
        [13, 17],
    ]
)

Y = X @ A

print(Y)