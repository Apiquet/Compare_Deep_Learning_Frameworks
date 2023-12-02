# Compare_Deep_Learning_Frameworks

Article [link](https://apiquet.com/2020/11/07/ssd300-implementation/) to read the full comparison between FastAI, JAX, Keras, MXNet, PaddlePaddle, Pytorch and Pytorch-lightning:

* ease of implementation (user friendly coding, ease of finding information online, etc.),
* time per epoch for the same model and the same training parameters,
* memory and GPU usage (thanks to pytorch-profiler),
* accuracy obtained after the same training.

# How to use the code

compare_framework_colab.ipynb notebook allows to run a CIFAR10 training using any frameworks. Only the framework_name variable should be updated to switch between frameworks. 

<a target="_blank" href="https://colab.research.google.com/github/Apiquet/Compare_Deep_Learning_Frameworks/blob/main/compare_framework_colab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
