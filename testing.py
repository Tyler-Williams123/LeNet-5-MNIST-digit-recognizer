import torch
import torchvision
import torchvision.transforms as transforms # figure out why we use different loss functions just all of this
import ConvNet

LeNet5 = ConvNet.convolutionalNetwork()

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor()
])

trainingData = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

dataLoader = torch.utils.data.DataLoader(trainingData, 64, shuffle=True)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(LeNet5.parameters())

# x= next(iter(dataLoader))[0]
# print(LeNet5(x))

for batch, (x, y) in enumerate(dataLoader):
    y_hat = LeNet5(x)

    y_true = torch.zeros(y.size(0), 10)
    y_true.scatter_(1, y.unsqueeze(1), 1)

    loss = loss_fn(y_hat, y_true)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if(batch % 50 == 0):
        print(loss.item())