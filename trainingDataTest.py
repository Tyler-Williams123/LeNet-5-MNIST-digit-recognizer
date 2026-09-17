import torch
import ConvNet

def main():
    device = torch.device("cuda")

    AlexNet = ConvNet.convolutionalNetwork().to(device)
    AlexNet.load_state_dict(torch.load("LeNet5.pt"))

    dataset = torch.utils.data.TensorDataset(*torch.load("data/MNIST_Preprocessed(MSE).pt"))
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    avg = []

    AlexNet.eval()
    with torch.no_grad():
        for x, y in dataLoader:
            x = x.to(device)
            y = torch.argmax(y.to(device), dim=1)

            guess = torch.argmax(AlexNet(x), dim=1)
            correct = sum((guess == y))
            avg.append(correct / y.shape[0])

    print(sum(avg) / len(avg))

if __name__ == "__main__":
    main()