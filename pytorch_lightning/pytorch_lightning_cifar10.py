import os

import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_pytorch_lightning_cifar10_data(batch_size=128) -> dict:
    """Pytorch lightling manages the loading of the database into the network class.

    This function is only present to be consistent with the code format
    of other frameworks.
    """
    return {}


class CNNModel(L.LightningModule):
    def __init__(
        self,
        batch_size: int = 128,
        lr: float = 0.0001,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.lr = lr
        # Set our init args as class attributes
        self.data_dir = os.environ.get("PATH_DATASETS", ".")

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (3, 32, 32)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ],
        )

        num_classes = 10
        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.train_db, self.val_db = random_split(dataset, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_db = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_db, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_db, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_db, batch_size=self.batch_size)


def run_pytorch_lightning_cifar10_training(
    dataloader: dict,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with pytorch_lightning frameworks.

    Returns:
        validation accuracy
    """
    model = CNNModel(batch_size=batch_size, lr=learning_rate)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        logger=CSVLogger(save_dir="logs/"),
    )
    trainer.fit(model)
    return trainer.test()["test_acc"]
