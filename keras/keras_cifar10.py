# Import necessary libraries
from tensorflow import keras


def get_data(batch_size=128) -> dict:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
       CIFAR10 dataloader
    """
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return {"train_data": (x_train, y_train), "test_data": (x_test, y_test)}


# Define the neural network
class CustomCNN(keras.Model):
    def __init__(self):
        super().__init__()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Conv2D(16, (3, 3), activation="relu"))
        self.model.add(keras.layers.MaxPooling2D((2, 2)))
        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(64, activation="relu"))
        self.model.add(keras.layers.Dense(10, activation="softmax"))

    def call(self, x):
        return self.model(x)


def run_training(
    dataloader: dict,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with keras frameworks.

    Returns:
        validation accuracy
    """
    (x_train, y_train), (x_test, y_test) = dataloader["train_data"], dataloader["test_data"]

    # Compile the model
    model = CustomCNN()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
    )

    # Evaluate the model
    _, test_acc = model.evaluate(x_test, y_test)
    return test_acc
