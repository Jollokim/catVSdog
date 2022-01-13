import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm



class convnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool3 = nn.MaxPool2d(2, 2)


        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)

        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))


        # x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
            print("lastDimout", x[0].shape)
            print("linear", self._to_linear)

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)

        x = F.relu(self.fc1(x))

        x = self.drop1(x)
        
        x = self.fc2(x)
        return F.softmax(x, dim=1)


net = convnet()