from fastai.data.core import DataLoaders
from fastai.vision.all import *


def get_fastai_cifar10_data(batch_size=128) -> DataLoaders:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
       CIFAR10 dataloader
    """
    path = untar_data(URLs.CIFAR)
    return ImageDataLoaders.from_folder(path, train="train", valid="test", bs=batch_size)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out


def run_fastai_cifar10_training(
    dataloader: DataLoaders,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with FASTAI frameworks.

    Returns:
        validation accuracy
    """
    model = CustomCNN()
    learn = Learner(dataloader, model, metrics=accuracy)

    learn.fit(epochs, lr=learning_rate)

    return learn.validate()[1]
