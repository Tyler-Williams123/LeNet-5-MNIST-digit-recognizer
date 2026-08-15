import ConvNet
import torchvision
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

testingData = torchvision.datasets.MNIST(
    "data",
    train=False,
    transform=transform,
    download=True
)

LeNet5 = ConvNet.convolutionalNetwork()
LeNet5.load_state_dict(torch.load("LeNet5.pt"))
dataLoader = torch.utils.data.DataLoader(testingData, 10)

totalCorrect = 0
total = 0

LeNet5.eval()
with torch.no_grad():
    for x, y in dataLoader:
        output = LeNet5(x)
        output = torch.argmax(output, 1)

        totalCorrect += (output == y).sum().item()
        total += y.size(0)

    accuracy = totalCorrect / total * 100
    print(accuracy)