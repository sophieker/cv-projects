import torch 
from torch import nn
import matplotlib.pyplot as plt 

weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * x + bias

print(x)
print(y)