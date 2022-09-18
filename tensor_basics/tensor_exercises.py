import torch 

rand_tensor = torch.rand(7, 7)
rand_tensor_2 = torch.rand(1, 7)

# print(rand_tensor)
# print(rand_tensor_2)

mult = torch.matmul(rand_tensor, rand_tensor_2.T)

# print(mult)

# set seed
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)

rand_tensor = torch.rand(7, 7)
rand_tensor_2 = torch.rand(1, 7)

mult = torch.matmul(rand_tensor, rand_tensor_2.T)
print(mult)