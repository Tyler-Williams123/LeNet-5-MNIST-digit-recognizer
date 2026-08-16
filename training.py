import time

import torch
import torchvision
import torchvision.transforms as transforms # figure out why we use different loss functions just all of this
import ConvNet

device = torch.device("cuda")

LeNet5 = ConvNet.convolutionalNetwork().to(device)

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
])

trainingData = torchvision.datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True,
)

dataLoader = torch.utils.data.DataLoader(trainingData, 64, shuffle=True)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(LeNet5.parameters())

torch.cuda.synchronize()
start = time.perf_counter()

for i in range (10):
    for batch, (x, y) in enumerate(dataLoader):
        x = x.to(device)
        y = y.to(device)

        y_hat = LeNet5(x)

        y_true = torch.zeros(y.size(0), 10, device=device)
        y_true.scatter_(1, y.unsqueeze(1), 1)

        loss = loss_fn(y_hat, y_true)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if(loss.item() < 0.003):
            break

        if(batch % 50 == 0):
            print(loss.item())

torch.cuda.synchronize()
end = time.perf_counter()

torch.save(LeNet5.state_dict(), "LeNet5.pt")