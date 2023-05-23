"""Script to profile multiple frameworks."""
from typing import Optional, Union

from torch import profiler


def get_framework_utils(framework_name: str):
    if framework_name == "fastai":
        from Compare_Deep_Learning_Frameworks.fastai.fastai_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "jax":
        from Compare_Deep_Learning_Frameworks.jax.jax_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "keras":
        from Compare_Deep_Learning_Frameworks.keras.keras_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "mxnet":
        from Compare_Deep_Learning_Frameworks.mxnet.mxnet_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "paddlepaddle":
        from Compare_Deep_Learning_Frameworks.paddlepaddle.paddlepaddle_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "pytorch":
        from Compare_Deep_Learning_Frameworks.pytorch.pytorch_cifar10 import (
            get_data,
            run_training,
        )
    elif framework_name == "pytorch_lightning":
        from Compare_Deep_Learning_Frameworks.pytorch_lightning.pytorch_lightning_cifar10 import (
            get_data,
            run_training,
        )
    else:
        raise ValueError(
            f"{framework_name} not in fastai, jax, keras, mxnet, paddlepaddle, pytorch, pytorch_lightning",
        )

    return get_data, run_training


def run_training(
    framework_name: str,
    training_params: dict[str, Union[int, float]],
    path_to_save_profiling_log: Optional[str] = None,
    profiler_with_stack: bool = False,
) -> float:
    """Run Pytorch profiler on framework training.

    Args:
      framework_name: name of the framework
      training_params: epochs, batch_size and learning_rate
      path_to_save_profiling_log: path to save the pytorch profiler log
        if None, profiling tool will not be used,
        if specified, a 1-epoch training will be ran to get a small profiling log.
      profiler_with_stack: record source information (file and line number) for the ops.

    Returns:
        val accuracy of the framework
    """
    get_data, run_training = get_framework_utils(framework_name)
    dataloader = get_data()

    if path_to_save_profiling_log is not None:
        prof = profiler.profile(
            schedule=profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=profiler.tensorboard_trace_handler(path_to_save_profiling_log),
            record_shapes=True,
            with_stack=profiler_with_stack,
        )
        training_params["epochs"] = 1
    if path_to_save_profiling_log is not None:
        print("Start profiler")
        prof.start()
    print("Start training")
    val_acc = run_training(dataloader=dataloader, **training_params)

    if path_to_save_profiling_log is not None:
        print("Stop and save profiler log")
        prof.step()
        prof.stop()

    return val_acc
