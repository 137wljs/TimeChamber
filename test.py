import torch

# 创建一个 4x3 的张量
tensor = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18],
                       [19, 20, 21],
                       [22, 23, 24]])
reshaped_tensor = tensor.view(2, 2 , 2, 3)

print(tensor)


print(reshaped_tensor)