import torch

batch_size = 4
num_channels = 3
shape = (100, 100)

y_true = torch.ones((batch_size, num_channels, *shape))
y = torch.ones((batch_size, num_channels, *shape))*0.9
prior = torch.ones((batch_size, 1, *shape))*0.2

res = y*prior

err = (y - y_true).pow(2) * prior

mean_e = err.mean()

print(mean_e)
