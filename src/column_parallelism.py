"""
src/column_parallelism.py
"""

import torch

X = torch.tensor(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
)

A1 = torch.tensor(
    [
        [10],
        [11],
        [12],
        [13],
    ]
)

A2 = torch.tensor(
    [
        [14],
        [15],
        [16],
        [17],
    ]
)

Y1 = X @ A1
Y2 = X @ A2

print(Y1)
print(Y2)

Y = torch.cat([Y1, Y2], dim=1)
print(Y)