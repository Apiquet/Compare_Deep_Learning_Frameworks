from typing import Tuple, Union

import tensorflow_datasets as tfds
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax import value_and_grad
from jax.example_libraries import optimizers, stax

conv_init, conv_apply = stax.serial(
    stax.Conv(32, (3, 3), padding="SAME"),
    stax.Relu,
    stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
    stax.Conv(16, (3, 3), padding="SAME"),
    stax.Relu,
    stax.MaxPool(window_shape=(2, 2), strides=(2, 2)),
    stax.Flatten,
    stax.Dense(64),
    stax.Relu,
    stax.Dense(10),
    stax.Softmax,
)


def get_jax_cifar10_data(batch_size=128) -> Union[tf.Tensor, tf.data.Dataset]:
    """Get DataLoaders for CIFAR-10 dataset.

    Returns:
        tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
    """
    cifar10_data, _ = tfds.load(
        name="cifar10",
        batch_size=batch_size,
        with_info=True,
    )
    return tfds.as_numpy(cifar10_data)


def MakePredictions(
    weights: list[Tuple[jax.Array, jax.Array]],
    input_data: jax.Array,
    batch_size: int,
) -> list[jax.Array]:
    """Make predictions for a batch of data.

    Args:
        weights: list from _, _, opt_get_weights = optimizers.adam(lr), opt_get_weights(opt_state)
        input_data: input data of shape (batch_size, width, height, channels)
        batch_size (int): The batch size.
    Returns:
        A list of predictions.
    """
    batches = jnp.arange((input_data.shape[0] // batch_size) + 1)  ### Batch Indices

    preds = []
    for batch in tqdm(batches, position=0, leave=True):
        if batch != batches[-1]:
            start, end = int(batch * batch_size), int(batch * batch_size + batch_size)
        else:
            start, end = int(batch * batch_size), None

        X_batch = input_data[start:end]

        if X_batch.shape[0] != 0:
            preds.append(conv_apply(weights, X_batch))

    return preds


def CrossEntropyLoss(
    weights: list,
    input_data: jax.Array,
    targets: jax.Array,
) -> jax.Array:
    """Implement of cross entropy loss.

    Args:
        weights: list from _, _, opt_get_weights = optimizers.adam(lr), opt_get_weights(opt_state)
        input_data: data to predict
        targets: groundtruth targets in one hot encoding

    Returns:
        loss value
    """
    preds = conv_apply(weights, input_data)
    log_preds = jnp.log(preds + tf.keras.backend.epsilon())
    return -jnp.mean(targets * log_preds)


def run_jax_cifar10_training(
    dataloader: Union[tf.Tensor, tf.data.Dataset],
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
) -> float:
    """Run CIFAR10 training with FASTAI frameworks.

    Returns:
        validation accuracy
    """
    train_data, test_data = dataloader["train"], dataloader["test"]
    X_train, Y_train = train_data["image"], train_data["label"]
    X_test, Y_test = test_data["image"], test_data["label"]

    X_train, X_test, Y_train, Y_test = (
        jnp.array(X_train, dtype=jnp.float32),
        jnp.array(X_test, dtype=jnp.float32),
        jnp.array(Y_train, dtype=jnp.float32),
        jnp.array(Y_test, dtype=jnp.float32),
    )
    rng = jax.random.PRNGKey(123)
    weights = conv_init(rng, (18, 32, 32, 3))[1]
    opt_init, opt_update, opt_get_weights = optimizers.adam(learning_rate)
    opt_state = opt_init(weights)
    Y_train_one_hot = jax.nn.one_hot(Y_train, num_classes=10)

    for i in range(epochs):
        batches = jnp.arange((X.shape[0] // batch_size) + 1)
        progress_bar = tqdm(batches, position=0, leave=True)

        losses = []
        for batch in batches:
            if batch != batches[-1]:
                start, end = int(batch * batch_size), int(batch * batch_size + batch_size)
            else:
                start, end = int(batch * batch_size), None

            X_batch, Y_batch = X_train[start:end], Y_train_one_hot[start:end]

            loss, gradients = value_and_grad(CrossEntropyLoss)(
                opt_get_weights(opt_state),
                X_batch,
                Y_batch,
            )

            ## Update Weights
            opt_state = opt_update(i, gradients, opt_state)

            losses.append(loss)

            progress_bar.set_description(f"Epoch {i+1}/{epochs}")
            progress_bar.set_postfix(train_loss=jnp.round(jnp.array(losses).mean(), decimals=3))
            progress_bar.update()

    test_preds = MakePredictions(opt_get_weights(opt_state), X_test, batch_size=batch_size)

    ## Combine predictions of all batches
    test_preds = jnp.concatenate(test_preds).squeeze()

    test_preds = jnp.argmax(test_preds, axis=1)
    return accuracy_score(Y_test, test_preds)
