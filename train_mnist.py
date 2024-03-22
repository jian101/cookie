import argparse
import os

import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import AlexNet
from torchvision.transforms import Compose, Resize, Grayscale, ToTensor
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()

parser.add_argument("--gpus", type=str, nargs="+", default=None)


def main():
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args.gpus)
        if torch.cuda.is_available():
            device = torch.device("cuda")

    model = AlexNet(num_classes=10).to(device)

    tr = Compose([Resize(size=(96, 96)), Grayscale(num_output_channels=3), ToTensor()])
    training_set = MNIST(root='./dataset', train=True, download=True, transform=tr)
    training_loader = DataLoader(dataset=training_set, batch_size=1024, shuffle=True, drop_last=False)

    test_set = MNIST(root='./dataset', train=False, download=True, transform=tr)
    test_loader = DataLoader(dataset=test_set, batch_size=512, shuffle=True, drop_last=False)

    loss_criterion = CrossEntropyLoss()

    optimizer = optim.AdamW(params=model.parameters(), lr=0.0001)

    def train_model(data_loader, model, optimizer, loss_criterion, device):
        model.train()
        model.to(device)
        for batch in tqdm(data_loader):
            img, target = batch
            img = img.to(device)
            target = target.to(device)
            outputs = model(img)
            loss = loss_criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epochs = 100
    for epoch in trange(epochs):
        train_model(training_loader, model, optimizer, loss_criterion, torch.device("cpu"))


if __name__ == "__main__":
    main()
