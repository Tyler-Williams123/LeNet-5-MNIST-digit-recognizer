import torch
import torch.nn as nn
import torch.nn.functional as F

class c3(nn.Module):
    def __init__(self,):
        super().__init__()
        self.connections = [
           (0, 1, 2),
           (1, 2, 3),
           (2, 3, 4),
           (3, 4, 5),
           (4, 5, 0),
           (5, 1, 2),
           (0, 1, 2, 3),
           (1, 2, 3, 4),
           (2, 3, 4, 5),
           (3, 4, 5, 0),
           (4, 5, 0, 1),
           (5, 0, 1, 2),
           (0, 1, 3, 4),
           (1, 2, 4, 5),
           (0, 2, 3, 5),
           (0, 1, 2, 3, 4, 5)
        ]

        self.connectionMap = torch.zeros(16, 6, 1, 1)

        for i in range(len(self.connections)):
            for link in self.connections[i]:
                self.connectionMap[i, link, :, :] = 1

        self.register_buffer("connectionMap", self.connectionMap)
        self.convLayer = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

    def forward(self, x):
        weights = self.convLayer.weight * self.connectionMap
        x = F.conv2d(x, weights, self.convLayer.bias)
        return x

class convolutionalNetwork(nn.Module):
    def __init__(self,):
        super().__init__()
        self.activationFunction = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.s2Weight = nn.Parameter(torch.randn(6))
        self.s2Bias = nn.Parameter(torch.randn(6))

        self.s4Weight = nn.Parameter(torch.randn(16))
        self.s4Bias = nn.Parameter(torch.randn(16))

        self.centers = nn.Parameter(torch.randn(10, 84))
        self.beta = nn.Parameter(torch.ones(10))

        self.layer1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.layer2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer3 = c3()
        self.layer4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.layer5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.f6 = nn.Linear(in_features=120, out_features=84)

    def forward(self, x):
        x = self.activationFunction(self.layer1(x))
        x = self.sigmoid(self.layer2(x) * self.s2Weight.view(1, 6, 1, 1) + self.s2Bias.view(1, 6, 1, 1))
        x = self.activationFunction(self.layer3(x))
        x = self.sigmoid(self.layer4(x) * self.s4Weight.view(1, 16, 1, 1) + self.s4Bias.view(1, 16, 1, 1))
        x = self.activationFunction(self.layer5(x))
        x = torch.flatten(x, 1)
        x = self.f6(x)
        distance = torch.cdist(x, self.centers)
        distance = distance ** 2
        x = torch.exp(-self.beta * distance)
        
        return x