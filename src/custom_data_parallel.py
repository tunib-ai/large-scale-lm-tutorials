"""
src/custom_data_parallel.py
"""

from torch import nn


# logits을 출력하는 일반적인 모델
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs


# forward pass에서 loss를 출력하는 parallel 모델
class ParallelLossModel(Model):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, labels):
        logits = super(ParallelLossModel, self).forward(inputs)
        loss = nn.CrossEntropyLoss(reduction="mean")(logits, labels)
        return loss
