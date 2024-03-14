# scrambled linear module.

import torch
import torch.nn as nn

class FuzzyLinear(nn.Module):
    def __init__(self, in_features, out_features, buffer, buffer_size, n_offsets):
        super(FuzzyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.buffer = buffer
        self.n_offsets = n_offsets
        self.offsets = torch.randint(0, buffer_size - in_features * out_features, (n_offsets, )).tolist()
        print(self.offsets)

    def forward(self, x):
        # pick one of the offsets for the weights?
        return torch.matmul(x, self.weight.t()) + self.bias

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


if __name__ == "__main__":
    fl = FuzzyLinear(16, 16, 100, 1024, 8)
    print(fl)
