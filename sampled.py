import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import random
from torch.nn.init import kaiming_uniform_
from pprint import pprint
from torch.optim import SGD

torch.manual_seed(1327)

def with_kaiming(out_features, in_features):
    weight = torch.empty(out_features, in_features)
    kaiming_uniform_(weight, nonlinearity='relu')
    return nn.Parameter(weight)

class SampledLinear(nn.Module):
    def __init__(self, weight_matrices):
        super(SampledLinear, self).__init__()

        # Store the weight matrices
        self.weight_matrices = weight_matrices
        self.in_features = self.weight_matrices[0].shape[1]
        self.out_features = self.weight_matrices[0].shape[0]
        self.selected_index = None

    def forward(self, x):
        # Randomly select one of the weight matrices
        if self.selected_index is None:
            self.selected_index = random.randint(0, len(self.weight_matrices) - 1)
        selected_weight = self.weight_matrices[self.selected_index]
        
        # Perform the matrix multiplication using the selected weight
        # Note: Bias is not implemented in this example, but you could easily add it
        return F.linear(x, selected_weight)
    
class ScrambledLinear(nn.Module):
    def __init__(self, weight_matrices):
        super(ScrambledLinear, self).__init__()

        self.slices = [SampledLinear(w) for w in weight_matrices]
        self.num_slices = len(self.slices)

    def forward(self, x):
        outputs = []
        for i in range(self.num_slices):
            slice_output = self.slices[i](x)
            outputs.append(slice_output)
        return torch.cat(outputs, dim=1)


in_features = 2
out_features = 2
number_of_matrices = 3

weight_matrices = nn.ParameterList([with_kaiming(out_features, in_features) for _ in range(number_of_matrices)])

pprint([w.data for w in weight_matrices])

l1 = SampledLinear(weight_matrices)
l2 = SampledLinear(weight_matrices)

s1 = ScrambledLinear([weight_matrices])

# in/out
input_tensor = torch.randn(1, in_features, requires_grad=True)
target = torch.randn(1, out_features)

optimizer = SGD(weight_matrices, lr=0.01)

optimizer.zero_grad()


# Forward pass
o1 = l1(input_tensor)
o2 = l2(input_tensor)

# Compute loss (MSE for simplicity)
loss = F.mse_loss(o1, target) + F.mse_loss(o2, target)

loss.backward()

print(f"Gradient of the selected weight matrix (Index {l1.selected_index}):")
print(l1.weight_matrices[l1.selected_index].grad)

print(f"Gradient of the selected weight matrix (Index {l2.selected_index}):")
print(l2.weight_matrices[l2.selected_index].grad)

optimizer.step()

pprint([w.data for w in weight_matrices])

optimizer.zero_grad()
o3 = s1(input_tensor)
loss = F.mse_loss(o3, target)
loss.backward()

print(f"Gradient of the selected weight matrix (Index {s1.slices[0].selected_index}):")
print(s1.slices[0].weight_matrices[s1.slices[0].selected_index].grad)