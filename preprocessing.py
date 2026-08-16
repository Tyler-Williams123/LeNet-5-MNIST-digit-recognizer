import torchvision
import torch
import torch.nn.functional as F

trainingData = torchvision.datasets.MNIST(
    "data",
    train=True,
    download=True
)

data = trainingData.data
targets = trainingData.targets

data = data.float() / 255.0
data = F.pad(data, (2, 2, 2, 2), value=0).unsqueeze_(1)
targets = torch.zeros(targets.size(0), 10).scatter_(1, targets.unsqueeze_(1), 1)

torch.save((data, targets), "MNIST_Preprocessed.pt")