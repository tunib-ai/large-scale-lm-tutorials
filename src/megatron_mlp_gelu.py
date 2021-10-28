"""
src/megatron_mlp_gelu.py
"""

import torch
from torch.nn.functional import gelu


w = torch.randn(6, 6)
x = torch.randn(6, 6)


class RowParallelLinear(torch.nn.Module):
    def __init__(self):
        super(RowParallelLinear, self).__init__()
        chunked = torch.chunk(w, 2, dim=0)

        # row parallelized parameters
        self.w1 = chunked[0]  # [3, 6]
        self.w2 = chunked[1]  # [3, 6]

    def forward(self, x):
        # GeLU(X1A1 + X2A2) != GeLU(X1A1) + GeLU(X2A2)
        x1, x2 = torch.chunk(x, 2, dim=1)

        # parallel output
        y1 = gelu(x1 @ self.w1) + gelu(x2 @ self.w2)

        # non-parallel output
        y2 = gelu(x1 @ self.w1 + x2 @ self.w2)

        return torch.all(y1 == y2)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self):
        super(ColumnParallelLinear, self).__init__()
        chunked = torch.chunk(w, 2, dim=1)

        # column parallelized parameters
        self.w1 = chunked[0]  # [6, 3]
        self.w2 = chunked[1]  # [6, 3]

    def forward(self, x):
        # GeLU(X1A1 cat X2A2) == GeLU(X1A1) cat GeLU(X2A2)

        # parallel output
        y1 = torch.cat([gelu(x @ self.w1), gelu(x @ self.w2)], dim=1)

        # non-parallel output
        y2 = gelu(torch.cat([(x @ self.w1), (x @ self.w2)], dim=1))

        return torch.all(y1 == y2)


# Row Parallelism
print("Is GeLU in RowParallelLinear same with non-parallel = ", end="")
print(RowParallelLinear()(x).item())

# Column Parallelism
print("Is GeLU in ColumnParallelLinear same with non-parallel = ", end="")
print(ColumnParallelLinear()(x).item())
