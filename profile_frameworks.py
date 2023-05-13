"""Script to profile multiple frameworks."""
import argparse
from typing import Union

from fastai_cifar10 import get_cifar10_data, run_fastai_cifar10_training
from google.colab import drive
from torch import profiler

drive.mount("/content/drive")
drive_folder_path = "/content/drive/MyDrive/Summaries/tests/compare_frameworks/"

IMPLEMENTED_FRAMEWORKS = {
    "fastai": {
        "get_data": get_cifar10_data,
        "run_training": run_fastai_cifar10_training,
    },
}


def profile_framework(
    framework_name: str,
    path_to_log: str,
    training_params: dict[str, Union[int, float]],
) -> float:
    """Run Pytorch profiler on framework training.

    Args:
      framework_name: name of the framework
      path_to_log: path to save the pytorch profiler log
      training_params: epochs, batch_size and learning_rate

    Returns:
        val accuracy of the framework
    """
    if framework_name not in IMPLEMENTED_FRAMEWORKS.keys():
        raise ValueError(f"{framework_name} not in {IMPLEMENTED_FRAMEWORKS.keys()}")

    dataloader = IMPLEMENTED_FRAMEWORKS[framework_name]["get_data"]()

    prof = profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(path_to_log),
        record_shapes=True,
        with_stack=True,
    )
    prof.start()
    val_acc = IMPLEMENTED_FRAMEWORKS[framework_name]["run_training"](
        dataloader=dataloader,
        **training_params,
    )
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
    args = parser.parse_args()
    training_params = {
        "epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.0001,
    }
    profile_framework(
        framework_name=args.framework_name,
        path_to_log=drive_folder_path,
        training_params=training_params,
    )
