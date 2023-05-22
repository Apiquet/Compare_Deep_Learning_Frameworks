"""Script to profile multiple frameworks."""
import argparse
from typing import Union

from torch import profiler


def get_framework_utils(framework_name: str):
    if framework_name == "fastai":
        from Compare_Deep_Learning_Frameworks.fastai.fastai_cifar10 import (
            get_fastai_cifar10_data,
            run_fastai_cifar10_training,
        )

        get_data = get_fastai_cifar10_data
        run_training = run_fastai_cifar10_training
    elif framework_name == "jax":
        from Compare_Deep_Learning_Frameworks.jax.jax_cifar10 import (
            get_jax_cifar10_data,
            run_jax_cifar10_training,
        )

        get_data = get_jax_cifar10_data
        run_training = run_jax_cifar10_training
    elif framework_name == "keras":
        from Compare_Deep_Learning_Frameworks.keras.keras_cifar10 import (
            get_keras_cifar10_data,
            run_keras_cifar10_training,
        )

        get_data = get_keras_cifar10_data
        run_training = run_keras_cifar10_training
    elif framework_name == "mxnet":
        from Compare_Deep_Learning_Frameworks.mxnet.mxnet_cifar10 import (
            get_mxnet_cifar10_data,
            run_mxnet_cifar10_training,
        )

        get_data = get_mxnet_cifar10_data
        run_training = run_mxnet_cifar10_training
    elif framework_name == "paddlepaddle":
        from Compare_Deep_Learning_Frameworks.paddlepaddle.paddlepaddle_cifar10 import (
            get_paddlepaddle_cifar10_data,
            run_paddlepaddle_cifar10_training,
        )

        get_data = get_paddlepaddle_cifar10_data
        run_training = run_paddlepaddle_cifar10_training
    elif framework_name == "pytorch":
        from Compare_Deep_Learning_Frameworks.pytorch.pytorch_cifar10 import (
            get_pytorch_cifar10_data,
            run_pytorch_cifar10_training,
        )

        get_data = get_pytorch_cifar10_data
        run_training = run_pytorch_cifar10_training
    elif framework_name == "pytorch_lightning":
        from Compare_Deep_Learning_Frameworks.pytorch_lightning.pytorch_lightning_cifar10 import (
            get_pytorch_lightning_cifar10_data,
            run_pytorch_lightning_cifar10_training,
        )

        get_data = get_pytorch_lightning_cifar10_data
        run_training = run_pytorch_lightning_cifar10_training
    else:
        raise ValueError(
            f"{framework_name} not in fastai, jax, keras, mxnet, paddlepaddle, pytorch, pytorch_lightning"
        )

    return get_data, run_training


def run_training(
    framework_name: str,
    path_to_log: str,
    training_params: dict[str, Union[int, float]],
    enable_profiling: bool = False,
) -> float:
    """Run Pytorch profiler on framework training.

    Args:
      framework_name: name of the framework
      path_to_log: path to save the pytorch profiler log
      training_params: epochs, batch_size and learning_rate

    Returns:
        val accuracy of the framework
    """
    get_data, run_training = get_framework_utils(framework_name)
    dataloader = get_data()

    if enable_profiling:
        prof = profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(path_to_log),
            record_shapes=True,
            with_stack=True,
        )
    if enable_profiling:
        print("Start profiler")
        prof.start()
    print("Start training")
    val_acc = run_training(dataloader=dataloader, **training_params)

    if enable_profiling:
        print("Stop and save profiler log")
        prof.step()
        prof.stop()

    return val_acc
