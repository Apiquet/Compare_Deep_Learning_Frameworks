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
    elif framework_name == "mxnet":
        from Compare_Deep_Learning_Frameworks.mxnet.mxnet_cifar10 import (
            get_mxnet_cifar10_data,
            run_mxnet_cifar10_training,
        )

        get_data = get_mxnet_cifar10_data
        run_training = run_mxnet_cifar10_training
    else:
        raise ValueError(f"{framework_name} not in fastai, jax and mxnet")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--framework_name",
        "-n",
        type=str,
        help="Framework to profile",
        choices=list(IMPLEMENTED_FRAMEWORKS.keys()),
    )
    parser.add_argument(
        "--path_to_log",
        "-o",
        type=str,
        help="Path to save profiler log",
        choices=list(IMPLEMENTED_FRAMEWORKS.keys()),
    )
    parser.add_argument(
        "--enable_profiling",
        "-p",
        type=bool,
        action="store_true",
        help="Profile framework training.",
        choices=list(IMPLEMENTED_FRAMEWORKS.keys()),
    )
    args = parser.parse_args()
    training_params = {
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.0001,
    }
    run_training(
        framework_name=args.framework_name,
        path_to_log=args.path_to_log,
        training_params=training_params,
        enable_profiling=args.enable_profiling,
    )
