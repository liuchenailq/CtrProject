"""
测试一些tensor的操作
"""

import torch

a = [torch.FloatTensor([[1,2], [1,2]]), torch.FloatTensor([[1,2], [1,2]])]
print(sum(a))
print(torch.sum(sum(a), 1))