import tensorflow as tf
import numpy as np
from quantazation import CENTERS

X = tf.random.normal(shape=(200, 2))
y = tf.concat([tf.zeros((100, 1)), tf.ones((100, 1))], axis=0)


def _transform():
    s = tf.cast(tf.reduce_sum(CENTERS), 'float32')
    return s


def transform(x, y):
    """

    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    n, _ = tf.shape
    s = _transform()
    return s * x, y


dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.map(lambda x, y: transform(x, y))
dataset = dataset.batch(2)
for x, y in dataset:
    print(x, " :: ", y)
