import torch
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# TENSOR = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(TENSOR.ndim)

# mask = torch.ones_like(input = TENSOR)
# for i in range(0, mask.size(dim=1)):
#     mask[i][0] = 0
# print(mask)

# print(TENSOR * mask)



# int_tensor = torch.tensor([1, 2, 3], dtype = torch.int8, device = "cpu", requires_grad = False)
# print(int_tensor)
# print(f"int tensor device: {int_tensor.device}")




# float_32_rand_tensor = torch.rand(size=(3, 4))
# float_16_rand_tensor = float_32_rand_tensor.type(torch.float16)
# print(f"type of float 16 tensor: {float_16_rand_tensor.dtype}")
# print(f"type of float 32 tensor: {float_32_rand_tensor.dtype}")
# long_tensor = float_16_rand_tensor.type(torch.long)
# print(f"type of long tensor: {long_tensor.dtype}")

# print(torch.matmul(torch.rand(7, 2), torch.rand(2, 4)))


# # practice squeezing
# original_tensor = torch.rand(size = (1, 1, 2, 3))
# print(original_tensor)
# print(f"shape of original tensor: {original_tensor.shape}")
# print(f"dimensions of original tensor: {original_tensor.ndim}")

# squeezed_tensor = torch.squeeze(original_tensor)
# print(squeezed_tensor)
# print(f"shape of squeezed tensor: {squeezed_tensor.shape}")
# print(f"dimensions of squeezed tensor: {squeezed_tensor.ndim}")

# # unsqueeze tensor
# unsqueezed_tensor = torch.unsqueeze(squeezed_tensor, 2)
# print(f"unsqueezed tensor shape: {unsqueezed_tensor.shape}")


# reshaping tensors
# original_tensor = torch.rand(size = (1, 10))
# print(original_tensor)
# reshaped_tensor = torch.reshape(original_tensor, (5, 2))
# print(reshaped_tensor)

# stack tensors
# x = torch.arange(1, 11, 1)
# stacked_tensor = torch.stack((x, x, x, x), dim = 0)
# print(stacked_tensor)

# print(f"shape of original and stacked tensor: {x.shape}, {stacked_tensor.shape}")


# permute tensors
# original = torch.rand(1, 4, 3)
# permuted = torch.permute(original, (1, 0, 2))
# print(f"original tensor size: {original.size()}")
# print(f"permuted tensor size: {permuted.size()}")

# practice indexing

# print(original[:, :, 1].shape)

# TENSOR = torch.arange(1, 10).reshape((1, 3, 3))
# print(TENSOR[:, :, 2])

# RANDOM_SEED = 42
# torch.manual_seed(RANDOM_SEED)
# x = torch.rand(3, 4)
# torch.manual_seed(RANDOM_SEED)
# y = torch.rand(3, 4)

# print(x == y)