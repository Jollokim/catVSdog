import torch
import torch.nn as nn

m = nn.Dropout(p=0.5)
input = torch.randn(4, 3)
output = m(input)

print(output)