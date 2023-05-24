from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_data(batch_size=128) -> dict:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
       CIFAR10 dataloader
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ],
    )

    # Load the training and test datasets
    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    return {"trainloader": trainloader, "testloader": testloader}


# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size: Tuple[int, int], classes_count: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        input_flatten_size = int(32 * (input_size[0] / 4) * (input_size[1] / 4))
        self.fc1 = nn.Linear(input_flatten_size, 64)
        self.fc2 = nn.Linear(64, classes_count)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def run_training(
    dataloader: dict,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with pytorch frameworks.

    Returns:
        validation accuracy
    """
    trainloader, testloader = dataloader["trainloader"], dataloader["testloader"]
    classes_count = 10
    for data in trainloader:
        input_data, _ = data
        input_size = input_data.shape

    net = Net(input_size[2:4], classes_count)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train the network
    for epoch in range(epochs):
        progress_bar = tqdm(trainloader, position=0, leave=True)
        running_loss = 0.0
        for data in trainloader:
            input_data, labels = data

            optimizer.zero_grad()

            outputs = net(input_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            progress_bar.set_postfix(train_loss=round(loss.item(), 3))
            progress_bar.update()

    predictions = []
    targets = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            predictions.append(outputs)
            targets.append(labels)

    predictions = torch.argmax(torch.cat(predictions, dim=0), dim=1)
    targets = torch.cat(targets, dim=0)
    return accuracy_score(targets, predictions)
