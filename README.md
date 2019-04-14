# Pytorch Light

Pytorch Light is a Python package that provides several tools about modelling and applications.

## Installation

- Installing from pypi: `pip install pytorchlight`

There are some features as follows:

- [Model](#model)
- [Data](#data)
- [Log](#log)
- [Visualization](#visualization)

[requirements.txt](requirements.txt) is also provided.

Next feature is going to be achieved as soon as possible.

- Adversarial Learning

And some applications are coming:

- Computer Vision
- Natural Language
- Rereinforcement Learning

**Note** that these features are only validated on cpu.

## Model

Application for model.

Features:

1. train
2. evaluate
3. batch loss and batch metric
4. model wrapper

## Data

Application for data.

Features:

1. Numpy.ndarray to/from Torch.tensor
2. Numpy.ndarray to/from Torch.Dataset
3. Numpy.ndarray to/from Torch.DataLoader

## Log

Application for log.

Features:

1. debug
2. info
3. warn
4. error
5. critical

## Visualization

Application for visualization based on numpy.ndarray.

Require:

- In the notebook mode, `%matplotlib inline` should be implement first.

Features:

1. learning curves
2. batch curves
3. one or multi images
