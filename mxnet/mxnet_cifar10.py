from tqdm import tqdm

from mxnet import autograd, context, cpu, gluon, gpu, init, nd, ndarray
from mxnet.gluon.data.vision import datasets, transforms

# Set the context to CPU or GPU
ctx = gpu() if context.num_gpus() > 0 else cpu()


def get_mxnet_cifar10_data(batch_size=128) -> dict:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
       CIFAR10 dataloader
    """
    # Load and transform the CIFAR10 data
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ],
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ],
    )
    train_data = datasets.CIFAR10(train=True).transform_first(transform_train)
    test_data = datasets.CIFAR10(train=False).transform_first(transform_test)
    return {"train_data": train_data, "test_data": test_data}


class CustomCNN:
    def __init__(self, num_classes=10):
        super().__init__()
        # Define the network architecture
        self.net = gluon.nn.Sequential()
        with self.net.name_scope():
            self.net.add(
                gluon.nn.Conv2D(channels=16, kernel_size=3, padding=1, activation="relu"),
                gluon.nn.MaxPool2D(pool_size=2, strides=2),
                gluon.nn.Conv2D(channels=32, kernel_size=3, padding=1, activation="relu"),
                gluon.nn.MaxPool2D(pool_size=2, strides=2),
                gluon.nn.Flatten(),
                gluon.nn.Dense(64, activation="relu"),
                gluon.nn.Dense(num_classes),
            )
        self.net.initialize(init=init.Xavier(), ctx=ctx)


def run_mxnet_cifar10_training(
    dataloader: dict,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with MXNET frameworks.

    Returns:
        validation accuracy
    """
    net = CustomCNN().net
    train_data, test_data = dataloader["train_data"], dataloader["test_data"]
    # Define the loss function and optimizer
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "adam", {"learning_rate": learning_rate})

    # Train the network
    batch_size = 128
    train_loader = gluon.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = gluon.data.DataLoader(test_data, batch_size)
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, position=0, leave=True)
        train_loss, train_acc, n = 0.0, 0.0, 0
        for X, y in train_loader:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                loss = softmax_cross_entropy(y_hat, y)
            loss.backward()
            trainer.step(batch_size)
            current_loss = nd.sum(loss).asscalar()
            train_loss += current_loss
            current_acc = nd.sum(
                ndarray.cast(y_hat.argmax(axis=1), dtype="int32") == y,
            ).asscalar()
            train_acc += current_acc
            n += y.size

            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            progress_bar.set_postfix(train_loss=round(current_loss, 3), train_acc=current_acc)
            progress_bar.update()

    test_acc = 0.0
    for X, y in test_loader:
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        y_hat = net(X)
        current_acc = nd.sum(
            ndarray.cast(y_hat.argmax(axis=1), dtype="int32") == y,
        ).asscalar()
        test_acc += current_acc

    test_acc /= len(test_data)
    return test_acc
