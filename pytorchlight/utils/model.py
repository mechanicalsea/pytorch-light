"""Application for model

Features:
    1. train
    2. evaluate
    3. batch loss and batch metric
    4. model wrapper

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from torch import nn
import torch.nn.functional as F
from .log import Log

__all__ = ['model_logger', 'Wrapper', 'Mnist_CNN', 'Mnist_Logistic',
           'batch_log', 'epoch_log',
           'loss_batch', 'metric_batch', 'run_train', 'run_eval']

model_logger = Log('model.py')


class Wrapper(nn.Module):
    """Wrapper wraps a model

    Construct a house to pack a model and provide additional features.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.info = False
        model_logger.info('Build model {}\n{}'
                          .format(type(model).__name__,
                                  model))

    def forward(self, x):
        if self.info:
            model_logger.info('{} forward ({}:{})'
                              .format(type(self.model).__name__,
                                      type(x).__name__, len(x)))
        y = self.model(x)
        return y

    def set_info(self, info):
        """Switch model log state.

        If message printing forward is heavy, keep info False which is also default.

        Args:
            info (boolean): False or True (default: {False})

        Example::

            >>> model = Minst_CNN()
            >>> wrapper = Wrapper(model)
            >>> wrapper.set_info(False)

        """
        self.info = info


class Mnist_CNN(nn.Module):
    """MNIST CNN model"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        return xb.view(-1, xb.size(1))


class Mnist_Logistic(nn.Module):
    """MNIST Linear model"""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Compute loss of model on a batch data

    loss_batch provide a way to compute model loss with mini-batch data.
    If optimizer exists, the weights of model parameter can be updated.

    Args:
        model (Model): a model with weights which can be updated
        loss_func (function): a function used to compute loss
        xb (tensor): features of a mini-batch data
        yb (tensor): labels of a mini-batch data corresponding to xb
        opt (optimizer): an optimizer (default: {None})

    Example::

        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch import optim
        >>> x = torch.tensor([[1, 2, 3], [3, 4, 5]])
        >>> y = torch.tensor([0, 1])
        >>> loss_func = F.cross_entropy
        >>> model = Mnist_CNN()
        >>> opt = optim.SGD(model.parameters(), lr=1e-3)
        >>> loss = loss_batch(model, loss_func, x, y, opt=opt)

    Returns:
        (float): a loss of model w.r.t the pair of xb and yb

    """
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item()


def metric_batch(model, metric_func, xb, yb):
    """Compute metric of model on a batch data.

    metric_batch provide a way to compute model metric with mini-batch data.
    The metric can be accuracy. It is necessary that metric_batch should be
    implemented under torch.no_grad().

    Args:
        model (Model): a model with weights which can be updated
        metric_func (function): a function used to compute metric
        xb (tensor): features of a mini-batch data
        yb (tensor): labels of a mini-batch data corresponding to xb

    Example::

        >>> import torch
        >>> x = torch.tensor([[1, 2, 3], [3, 4, 5]])
        >>> y = torch.tensor([0, 1])
        >>> def accuracy(out, y):
        >>>     preds = torch.argmax(out, dim=1)
        >>>     return (preds == y).float().mean()
        >>> model = Mnist_CNN()
        >>> with torch.no_grad():
        >>>     acc = metric_batch(model, accuracy, x, y)

    Returns:
        (float): a metric of model w.r.t the pair of xb and yb

    """
    metric = metric_func(model(xb), yb)
    return metric.item()


def batch_log(loss, loss_func, nums, n_steps=None,
              opt=None, metric=None, metric_func=None):
    """Log for batch while training.

    Args:
        loss (float): loss of batch
        loss_func (function): loss function
        nums (int): size of batch
        n_steps (tuple(int, int)): step out of total steps (default: {None})
        opt (optimizer): optimizer type (default: {None})
        metric (float): metric of batch (default: {None})
        metric_func (function): metric function (default: {None})

    Example::

        >>> import torch.nn.functional as F
        >>> from torch import optim
        >>> loss_func = F.cross_entropy
        >>> def accuracy(out, y):
        >>>     preds = torch.argmax(out, dim=1)
        >>>     return (preds == y).float().mean()
        >>> model = Mnist_CNN()
        >>> opt = optim.SGD(model.parameters(), lr=1e-3)
        >>> batch_log(0.1, loss_func, 64, n_steps=(1, 100),
                      opt=opt, metric=0.2, metric_func=accuracy)

    """
    step = '' if n_steps is None else '%4d/%4d - ' % n_steps
    optimizer = '' if opt is None else ' in %s' % type(opt).__name__
    if metric is None:
        model_logger.debug('%s%s: %.4f - %d%s.'
                           % (step,
                              loss_func.__name__, loss,
                              nums, optimizer))
    else:
        model_logger.debug('%s%s: %.4f - %s: %.4f - %d%s.'
                           % (step,
                              loss_func.__name__, loss,
                              metric_func.__name__, metric,
                              nums, optimizer))


def epoch_log(loss, loss_func, nums=None, epoch=None, mode='Train',
              metric=None, metric_func=None):
    """Log for epoch train or validation.

    Args:
        loss (float): loss of batch
        loss_func (function): loss function
        nums (list[int]): a series of batch size (default: {None})
        epoch (int): the number of epoch (default: {None})
        mode (str): 'Train' or 'Valid' (default: {'Train'})
        metric (float): metric of batch (default: {None})
        metric_func (function): metric function (default: {None})

    Example::

        >>> import torch.nn.functional as F
        >>> loss_func = F.cross_entropy
        >>> def accuracy(out, y):
        >>>     preds = torch.argmax(out, dim=1)
        >>>     return (preds == y).float().mean()
        >>> epoch_log(0.1, loss_func, 
                      nums=[64, 64, 16], epoch=1, mode='Train',
                      metric=0.2, metric_func=accuracy)

    """
    E = '' if epoch is None else 'Epoch %3d' % epoch
    N = '' if nums is None else '/%d - ' % np.sum(nums)
    if metric is None:
        model_logger.debug(mode + ' - ' + E + N + '%s: %.4f' % (
            loss_func.__name__, loss))
    else:
        model_logger.debug(mode + ' - ' + E + N + '%s: %.4f, %s: %.4f' % (
            loss_func.__name__, loss,
            metric_func.__name__, metric))


def run_train(epochs, model, loss_func, opt, train_dl,
              valid_dl=None, metric_func=None):
    """Train model with given trainning data and loss function.

    Args:
        epochs (int): total train epoch
        model (Model): training model
        loss_func (function): loss function
        opt (optimizer): optimizer
        train_dl (DataLoader): data loader for training
        valid_dl (DataLoader): data loader for validation (default: {None})
        metric_func (function): metric function (default: {None})

    Example::

        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch import optim
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> loss_func = F.cross_entropy
        >>> def accuracy(out, y):
        >>>     preds = torch.argmax(out, dim=1)
        >>>     return (preds == y).float().mean()
        >>> model = Mnist_CNN()
        >>> opt = optim.SGD(model.parameters(), lr=1e-3)
        >>> tds = TensorDataset(torch.tensor([[1, 2, 3, 4, 5],
                                              [5, 6, 7, 8, 9]]),
                                torch.tensor([0, 1]))
        >>> train_dl = DataLoader(tds)
        >>> valid_dl = DataLoader(tds)
        >>> run_train(5, model, loss_func, opt, train_dl, 
                      valid_dl=valid_dl, metric_func=accuracy)

    """
    for epoch in range(max(1, epochs)):
        model.train()
        epoch_loss, nums, epoch_metric = [], [], []
        for step, (xb, yb) in enumerate(train_dl):
            mini_loss = loss_batch(model, loss_func, xb, yb, opt)
            epoch_loss.append(mini_loss)
            if metric_func is not None:
                mini_metric = metric_batch(model, metric_func, xb, yb)
                epoch_metric.append(mini_metric)
            else:
                mini_metric = None
            nums.append(len(xb))
            batch_log(mini_loss, loss_func, len(xb),
                      n_steps=(step + 1, len(train_dl)),
                      opt=opt, metric=mini_metric, metric_func=metric_func)
        train_loss = np.sum(np.multiply(epoch_loss, nums)) / np.sum(nums)
        if metric_func is not None:
            train_metric = np.sum(np.multiply(
                epoch_metric, nums)) / np.sum(nums)
        else:
            train_metric = None
        epoch_log(train_loss, loss_func,
                  nums=nums, epoch=epoch + 1, mode='Train',
                  metric=train_metric, metric_func=metric_func)
        if valid_dl is not None:
            valid_loss, valid_metric = run_eval(model, loss_func,
                                                valid_dl, metric_func)
            epoch_log(valid_loss, loss_func,
                      epoch=epoch + 1, mode='Valid',
                      metric=valid_metric, metric_func=metric_func)


def run_eval(model, loss_func, valid_dl, metric_func=None):
    """Evaluate model with given validation data.

    Args:
        model (Model): training model
        loss_func (function): loss function
        valid_dl (DataLoader): data loader for validation (default: {None})
        metric_func (function): metric function (default: {None})

    Example::

        >>> import torch
        >>> import torch.nn.functional as F
        >>> from torch.utils.data import TensorDataset, DataLoader
        >>> loss_func = F.cross_entropy
        >>> def accuracy(out, y):
        >>>     preds = torch.argmax(out, dim=1)
        >>>     return (preds == y).float().mean()
        >>> model = Mnist_CNN()
        >>> tds = TensorDataset(torch.tensor([[1, 2, 3, 4, 5],
                                              [5, 6, 7, 8, 9]]),
                                torch.tensor([0, 1]))
        >>> valid_dl = DataLoader(tds)
        >>> run_eval(model, loss_func, valid_dl,
                     metric_func=accuracy)

    Returns:
        (tuple(float, float)): a tuple of loss and metric

    """
    model.eval()
    with torch.no_grad():
        loss, nums = zip(
            *[(loss_batch(model, loss_func, xb, yb), len(xb))
              for xb, yb in valid_dl]
        )
        if metric_func is not None:
            metric = tuple(
                metric_batch(model, metric_func, xb, yb)
                for xb, yb in valid_dl
            )
    valid_loss = np.sum(np.multiply(loss, nums)) / np.sum(nums)
    if metric_func is not None:
        valid_metric = np.sum(np.multiply(metric, nums)) / np.sum(nums)
    else:
        valid_metric = None
    return valid_loss, valid_metric
