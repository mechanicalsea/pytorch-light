"""Application for data

Features:
    1. Numpy.ndarray to/from Torch.tensor
    2. Numpy.ndarray to/from Torch.Dataset
    3. Numpy.ndarray to/from Torch.DataLoader

"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from .log import Log

__all__ = ['data_logger', 'n2t', 't2n', 'n2tds',
           'tds2n', 'n2dl', 'dl2n']

data_logger = Log('data.py')


def n2t(*arrays):
    """Convert numpy.ndarray(s) to torch.tensor(s).

    ndarray -> tensor

    Args:
        *arrays (ndarray): a tuple of data

    Example::

        >>> import numpy as np
        >>> a = np.random.rand(10)
        >>> b = np.random.rand(10)
        >>> ta, tb = n2t(a, b)

    Returns:
        (*tensor): a list of data with the same length of arrays

    """
    data_logger.info('n2t({}:{}{})'
                     .format(type(arrays).__name__, len(arrays),
                             tuple(type(a).__name__ for a in arrays)))
    if len(arrays) == 1:
        return list(map(torch.tensor, (arrays[0],)))[0]
    return list(map(torch.tensor, arrays))


def t2n(*tensors):
    """Convert torch.tensor(s) to numpy.ndarray(s).

    tensor -> ndarray

    Args:
        *tensors (tensor): a tuple of data

    Example::

        >>> import torch
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5, 6])
        >>> na, nb = t2n(a, b)

    Returns:
        (*ndarray): a list of data with the sample length of tensors

    """
    data_logger.info('t2n({}:{}{})'
                     .format(type(tensors).__name__, len(tensors),
                             tuple(type(t).__name__ for t in tensors)))
    if len(tensors) == 1:
        return list(map(np.array, (tensors[0],)))[0]
    return list(map(np.array, tensors))


def n2tds(*arrays):
    """Convert numpy.ndarray(s) to torch.utils.data.TensorDataset.

    ndarray -> TensorDataset

    Args:
        *arrays (ndarray): a tuple of data

    Example::

        >>> import numpy as np
        >>> a = np.random.rand(10)
        >>> b = np.random.rand(10)
        >>> tds = n2tds(a, b)

    Returns:
        (TensorDataset): a dataset with a series of arrays

    """
    data_logger.info('n2tds({}:{}{})'
                     .format(type(arrays).__name__, len(arrays),
                             tuple(type(a).__name__ for a in arrays)))
    return TensorDataset(*n2t(*arrays))


def tds2n(tds):
    """Convert torch.utils.data.TensorDataset to numpy.ndarray(s).

    TensorDataset -> ndarray

    Args:
        tds (TensorDataset): a dataset which consists of a series of arrays

    Example::

        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> tds = TensorDataset(torch.tensor([1, 2, 3])
                                torch.tensor([4, 5, 6]))
        >>> a, b = tds2n(tds)

    Returns:
        (*ndarray): a list of data with a series of arrays

    """
    data_logger.info('tds2n({})'.format(type(tds).__name__))
    return t2n(*tds.tensors)


def n2dl(*arrays, **options):
    """Convert numpy.array(s) to torch.utils.data.DataLoader

    ndarray -> DataLoader

    Args:
        *arrays (ndarray): a tuple of data
        **options (dict): a dict of DataLoader's parameters

    Example::

        >>> import numpy as np
        >>> a = np.random.rand(10)
        >>> b = np.random.rand(10)
        >>> dl = n2dl(a, b, shuffle=True, batch_size=2)

    Returns:
        (DataLoader): a dataloader for generation of arrays

    """
    data_logger.info('n2dl({}:{}{}, {})'
                     .format(type(arrays).__name__, len(arrays),
                             tuple(type(a).__name__ for a in arrays), options))
    return DataLoader(n2tds(*arrays), **options)


def dl2n(dl, indices=None):
    """Sample numpy.array(s) based on index from torch.utils.data.DataLoader

    dl -> ndarray

    Args:
        dl (DataLoader): a data loader
        indices (list): indices within a data loader (default: {None})

    Example::

        >>> import torch
        >>> from import.utils.data import TensorDataset, DataLoader

        >>> dl = DataLoader(TensorDataset(torch.tensor([1, 2, 3, 4, 5]),
                                          torch.tensor([5, 6, 7, 8, 9])),
                            shuffle=True, batch_size=2)
        >>> a, b = dl2n(dl, indices=[0, 1, 2])

    Returns:
        (list of ndarray): a series of data

    """
    data_logger.info('n2dl({}, {})'.format(type(dl).__name__, indices))
    indices = [0] if indices is None else indices
    return [d[indices] for d in dl.dataset.tensors]

