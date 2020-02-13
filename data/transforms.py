from typing import Tuple

import tensorflow as tf
from config import NUM_CLASSES_Q, CENTERS
from utils import rgb_to_lab, get_ab_channels, get_lightness_channel
from data.augmenters import rgb_to_gray

def quantize(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get a quantized version of a, b channels for  y image in
    LAB color space
    Parameters
    ----------
    x: tf.Tensor
        l_channel
    y: tf.Tensor
        ab_channels

    Returns
    -------
    x: l_channel
    y: quantized ab_channels

    """
    h, w, _ = y.shape
    y = tf.reshape(y, (-1, 2))
    distances, indices = nearest_neighbour(centers=CENTERS, sample=y, k=5,
                                           n=h * w)
    # smooth the distances with a gaussian kernel
    gauss_sigma = 5
    distances = tf.exp(-distances ** 2 / (2 * gauss_sigma ** 2))
    # Normalize the distances to get probability distribution
    distances = distances / tf.expand_dims(
        tf.reduce_sum(distances, axis=1), -1)

    # get shape of indices
    idx_sahpe = tf.shape(indices, out_type=indices.dtype)
    n = idx_sahpe[0]
    k = idx_sahpe[1]
    idx_row = tf.tile(tf.expand_dims(tf.range(n), 1), (1, k))
    idx_full = tf.stack([idx_row, indices], axis=-1)
    target = tf.scatter_nd(idx_full, distances, [n, NUM_CLASSES_Q])

    target = tf.reshape(target, (h, w, NUM_CLASSES_Q))
    return x, target


@tf.function
def nearest_neighbour(centers: tf.Tensor, sample: tf.Tensor, k, n) -> Tuple[
    tf.Tensor, tf.Tensor]:
    """
    Get k centers Nearest neighbour of the sample
    Parameters
    ----------
    centers: tf.Tensor
    sample: tf.Tensor
    k: int
        k nearest neighbors
    n: sample shape (w*h)

    Returns
    -------
    distances : tf.Tensor
    indx: tf.Tensor
    """
    # X^2
    centers_sqr = tf.expand_dims(tf.reduce_sum(tf.pow(centers, 2), axis=1),
                                 axis=0)
    # Y^2
    sample_sqr = tf.expand_dims(tf.reduce_sum(tf.pow(sample, 2), axis=1),
                                axis=1)

    # Repeat X^2 n times vertically
    centers_sqrR = tf.repeat(centers_sqr, n, axis=0)

    # Repeat Y^2 m times
    sample_sqrR = tf.repeat(sample_sqr, NUM_CLASSES_Q, axis=1)

    # X*Y
    cross = tf.matmul(sample, tf.transpose(centers))

    # X^2 + Y^2 - 2*X*Y
    distances = tf.sqrt(centers_sqrR + sample_sqrR - 2 * cross)

    # Return top_k distances and indices
    dist, indx = tf.math.top_k(-1 * distances, k=k, sorted=False)
    return -1 * dist, indx


def get_l_and_ab_channels(image: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Transform image from RGB to lab and return
    L and ab channels
    Parameters
    ----------
    image: tf.tensor
        input image

    Returns
    -------
    tuple(tf.Tensor, tf.Tensor)
    """
    gray = rgb_to_gray(image) / 255.0
    lab = rgb_to_lab(image)
    l_channel = get_lightness_channel(lab)
    ab_channels = get_ab_channels(lab)
    ab_channels = ab_channels - 128.0
    return gray, ab_channels
