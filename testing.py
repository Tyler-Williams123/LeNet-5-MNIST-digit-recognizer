import torch
import ConvNet

network = ConvNet.convolutionalNetwork()

nothing = torch.zeros((1, 1, 32, 32))

print(network(nothing).shape)