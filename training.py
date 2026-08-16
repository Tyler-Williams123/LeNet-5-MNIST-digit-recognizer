import time

import torch
import torchvision
import torchvision.transforms as transforms # figure out why we use different loss functions just all of this, figure out multicore jank
import ConvNet

device = torch.device("cuda")

def main():
    LeNet5 = ConvNet.convolutionalNetwork().to(device)

    dataset = torch.utils.data.TensorDataset(*torch.load("data/MNIST_Preprocessed(MSE).pt"))

    dataLoader = torch.utils.data.DataLoader(dataset, 240, shuffle=True, num_workers=2)
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(LeNet5.parameters())

    torch.cuda.synchronize()
    start = time.perf_counter()

    for i in range (10):
        for batch, (x, y) in enumerate(dataLoader):
            x = x.to(device)
            y = y.to(device)

            y_hat = LeNet5(x)

            loss = loss_fn(y_hat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if(loss.item() < 0.00005):
                torch.cuda.synchronize()
                end = time.perf_counter()
                print("converged to 0.00005 by " + str(end - start))

            if(batch % 50 == 0):
                print(loss.item())

    torch.save(LeNet5.state_dict(), "LeNet5.pt")

if __name__ == "__main__":
    main()