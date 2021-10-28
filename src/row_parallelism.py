"""
src/row_parallelism.py
"""

import torch

X1 = torch.tensor(
    [
        [0, 1],
        [4, 5],
    ]
)

X2 = torch.tensor(
    [
        [2, 3],
        [6, 7],
    ]
)

A1 = torch.tensor(
    [
        [10, 14],
        [11, 15],
    ]
)

A2 = torch.tensor(
    [
        [12, 16],
        [13, 17],
    ]
)

Y1 = X1 @ A1
Y2 = X2 @ A2

print(Y1)
print(Y2)

Y = Y1 + Y2
print(Y)