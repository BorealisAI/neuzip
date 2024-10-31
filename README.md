# NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks

This is the official repository for the paper [*NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks*](https://arxiv.org/abs/2410.20650). This repository contains the code for the experiments in the paper.

# Installation

## From PyPI

We provide a simple way to install the package from PyPI. You can install the package by running the following command:

```bash
pip install neuzip
```

Note that we only upload the source distribution to PyPI. You need to have NVCC correctly installed on your system to compile the package.

## From source

You can also install the package from source, which is useful if you want to modify the code.

```bash
git clone https://github.com/BorealisAI/neuzip
cd neuzip
pip install -e .
```

# Basci usage

Using `neuzip` for your PyTorch model is pretty easy. Here is a simple example:

```diff
model: torch.nn.Module = # your model
+ manager = neuzip.Manager()
+ model = manager.convert(model)
```

The compressed model can be used in the same way as the original model while consuming less memory.

# Replicating experiments

You can replicate all the experiments in the paper by using the files in the [examples/](examples/) directory. Each file corresponds to one or more experiments in the paper.
