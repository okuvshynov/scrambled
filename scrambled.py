# scrambled linear module.

import torch
import torch.nn as nn

class ScrambledLinear(nn.Module):
    def __init__(self, in_features, out_features, num_slices):
        super(ScrambledLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_slices = num_slices

        if out_features % num_slices != 0:
            raise ValueError(f"out_features must be divisible by num_slices. Got out_features={out_features} and num_slices={num_slices}")

        self.slice_size = out_features // num_slices
        self.slices = nn.ModuleList()

        for _ in range(num_slices):
            self.slices.append(nn.Linear(in_features, self.slice_size))

    def forward(self, x):
        outputs = []
        for i in range(self.num_slices):
            slice_output = self.slices[i](x)
            outputs.append(slice_output)
        return torch.cat(outputs, dim=1)

