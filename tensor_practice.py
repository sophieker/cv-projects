import torch

RANDOM_SEED = 1234
torch.cuda.manual_seed(RANDOM_SEED)
x = torch.rand(size=(2, 3), device = 'cuda:0')

torch.cuda.manual_seed(RANDOM_SEED)
y = torch.rand(size=(2, 3), device = 'cuda:0')

mult = torch.matmul(x, y.T)
print(mult)
print(f"max: {mult.max()}")

print(f"min: {mult.min()}")

z = torch.rand(size = (1, 1, 1, 10), device = 'cuda:0')
z_squeezed = z.squeeze()

print(f"first shape: {z.shape}")
print(f"second shape: {z_squeezed.shape}")
