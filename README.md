# NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks

This is the official repository for the paper "NeuZip: Memory-Efficient Training and Inference with Dynamic Compression of Neural Networks".
This repository contains the code for the experiments in the paper.

# Installation

First, please install NVComp library on your own. You can find the installation instructions on its [GitHub page](https://github.com/NVIDIA/nvcomp).

Then, you can install the package by running the following command in the root directory of the repository:

```bash
pip install -e .
```

# Usage

You can use the `neuzip` package to compress and decompress tensors. Here is an example:

```python
import neuzip
manager = neuzip.Manager()  # Create a manager
handle = manager.write(tensor)  # Compress a tensor
tensor = manager.read(handle)  # Decompress a tensor
```

# Replicating Experiments

You can replicate all the experiments in the paper by using the files in the [examples/](examples/) directory. Each file corresponds to one or more experiments in the paper.
