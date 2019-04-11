"""Application for visualization based on numpy.ndarray

Require:
    %matplotlib inline

Features:
    1. learning curves
    2. batch curves
    3. one or multi images

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from log import Log

__all__ = ['visual_logger', 'imshow', 'batch_curves', 'learning_curves',
           'oneshow']

visual_logger = Log('visual.py')


def imshow(arrays, mode='rgb'):
    """Grid view of pictures with RGB or GRAY.

    ndarray -> image

    Args:
        arrays (ndarray): RGB image(s) or GREY image(s)
        mode ('rgb' or 'grey'): image mode (default: {'rgb'})

    Example::

        >>> import numpy as np
        >>> image = np.random.rand(10, 10)
        >>> imshow(image, mode='grey')
        >>> images = np.random.rand(8, 10, 10)
        >>> imshow(images, mode='grey')
        >>> image = np.random.rand(10, 10, 3)
        >>> imshow(image)
        >>> image = np.random.rand(3, 10, 10)
        >>> imshow(image)
        >>> images = np.random.rand(8, 3, 10, 10)
        >>> imshow(images)

    Raises:
        TypeError: not isinstance(arrays, ndarray)
        ValueError: neither mode is 'rgb' nor 'grey'

    """
    visual_logger.info('imshow({}, {})'.format(type(arrays).__name__, mode))
    if not isinstance(arrays, np.ndarray):
        visual_logger.error(
            'TypeError - Invalid Type of arrays: {}'.format(
                type(arrays).__name__))
        raise TypeError('Invalid Type of arrays: {}'.format(
            type(arrays).__name__))
    if mode is not 'rgb' and mode is not 'grey':
        visual_logger.error('Invalid mode: %s' % mode)
        raise ValueError('Invalid mode: %s' % mode)
    if mode == 'rgb' and arrays.ndim == 3:
        if arrays.shape[0] == 3:
            arrays = arrays.transpose(1, 2, 0)
        elif arrays.shape[2] == 3:
            pass
        else:
            visual_logger.error(
                'TypeError - Invalid arrays with unrecognized channal: {}'
                .format(arrays.shape))
            raise TypeError(
                'Invalid arrays with unrecognized channal: {}'.format(
                    arrays.shape))
    # Figure's mode and size
    ndim = 3 if mode is 'rgb' else 2
    if arrays.ndim == ndim or (arrays.ndim == ndim + 1
                               and arrays.shape[0] == 1):
        plt.figure(figsize=(3, 3))
    elif arrays.ndim == ndim + 1:
        plt.figure(figsize=(8, 8))
    else:
        visual_logger.error('ValueError - Invalid ndim: %d(%s)' %
                            (arrays.ndim, mode))
        raise ValueError('Invalid ndim: %d(%s)' % (arrays.ndim, mode))
    # Figure grid or not
    if mode == 'grey' and arrays.ndim == 2:
        arrays = arrays[np.newaxis, :, :]
    if mode == 'grey' and arrays.ndim == 3:
        arrays = arrays[:, np.newaxis, :, :]
        arrays = torchvision.utils.make_grid(torch.tensor(arrays)).numpy()
        arrays = arrays.transpose(1, 2, 0)
    elif mode == 'rgb' and arrays.ndim == 4:
        if arrays.shape[3] == 3:
            arrays = arrays.transpose(0, 3, 1, 2)
        arrays = torchvision.utils.make_grid(torch.tensor(arrays)).numpy()
        arrays = arrays.transpose(1, 2, 0)
    plt.imshow(arrays)
    plt.show()


def batch_curves(loss, acc=None, beta=0.9):
    """Figure describes single loss and single accuracy.

    Args:
        loss (ndarray, list): one-dimension-vector loss
        acc (ndarray, list): one-dimension-vector accuracy (default: {None})
        beta (float): smoothing factor of curves, [0, 1] (default: {0.9})

    Example::

        >>> import numpy as np
        >>> loss = np.random.rand(100)
        >>> batch_curves(loss)
        >>> acc = np.random.rand(100)
        >>> batch_curves(loss, acc)

    """
    visual_logger.info('batch_curves({}, {}, {})'
                       .format(type(loss).__name__, type(acc).__name__, beta))
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111)
    t = np.arange(len(loss))
    ax1.plot(t, loss, 'orange', alpha=0.3)
    for i in range(1, len(loss)):
        loss[i] = (1. - beta) * loss[i] + beta * loss[i - 1]
    ax1.plot(t, loss, 'orange')
    ax1.set_ylabel('Loss')
    if acc is not None:
        ax2 = ax1.twinx()  # this is the important function
        t = np.arange(len(acc))
        ax2.plot(t, acc, 'r', alpha=0.3)
        for i in range(1, len(acc)):
            acc[i] = (1. - beta) * acc[i] + beta * acc[i - 1]
        ax2.plot(t, acc, 'r')
        ax2.set_ylabel('Accuracy')
    ax1.set_xlabel('Batch')
    ax1.set_xlim([0, max(t)])
    plt.show()


def learning_curves(loss, val_loss, acc, val_acc):
    """Figure describes learning curves.

    Args:
        loss (ndarray, list): one-dimension-vector train loss
        val_loss (ndarray, list): one-dimension-vector valid loss
        acc (ndarray, list): one-dimension-vector train accuracy
        val_acc (ndarray, list): one-dimension-vector train accuracy

    Example::

        >>> import numpy as np
        >>> loss = np.random.rand(100)
        >>> val_loss = np.random.rand(100)
        >>> acc = np.random.rand(100)
        >>> val_acc = np.random.rand(100)
        >>> learning_curves(loss, val_loss, acc, val_acc)

    """
    visual_logger.info('learning_curves({}, {}, {}, {})'
                       .format(type(loss).__name__, type(val_loss).__name__,
                               type(acc).__name__, type(val_acc).__name__))

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlim([0, len(loss)])
    plt.ylim([0, max(plt.ylim())])
    plt.title('Training and Validation Loss')
    plt.subplot(1, 2, 2)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xlim([0, len(acc)])
    plt.ylim([0, 1])
    plt.title('Training and Validation Accuracy')
    plt.show()


def oneshow(image_names):
    """Show a image or multi images, such as JEPG.

    This function is accessiable to image paths.

    Args:
        image_names (list, str):  a image file or a set of image files

    Example::

        >>> oneshow('1.jpg')
        >>> oneshow(['1.jpg', '2.jpg'])

    Raises: 
        TypeError: neither image_names is str nor list of str
        FileNotFoundError: all of image files can not be found. 
    """
    if not isinstance(image_names, (str, list)):
        visual_logger.error(
            'TypeError - Invalid image files: {}'.format(image_names))
        raise TypeError('Invalid image files: {}'.format(image_names))
    if isinstance(image_names, str):
        if not os.path.exists(image_names):
            visual_logger.error(
                'FileNotFoundError - Missing a image file: {}'
                .format(image_names))
            raise FileNotFoundError(
                'Missing a image file: {}'.format(image_names))
        image = plt.imread(image_names)
    else:
        image_miss = [img for img in image_names if not os.path.exists(img)]
        if len(image_miss) > 0:
            if len(image_miss) < len(image_names):
                visual_logger.warn('Missing {} image files: {}'.format(
                    len(image_miss), image_miss))
            else:
                visual_logger.error(
                    'FileNotFoundError - Missing all image files: {}'
                    .format(image_miss))
                raise FileNotFoundError(
                    'Missing all image file: {}'.format(image_miss))
        image = np.array([plt.imread(img)
                          for img in image_names if os.path.exists(img)])
    imshow(image)
