import tensorflow as tf
import numpy as np


# all functions are implemented based on
# <http://www.easyrgb.com/en/math.php#text8>


def lab_to_xyz(l: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Convert CIELAB image into XYZ image
    Parameters
    ----------
    l: np.array
        L channel
    a: np.array
        a channel
    b : np.array
        b channel

    Returns
    -------
    np.array: image in the XYZ color space
    """
    var_y = (l * 16) / 116
    var_x = a / 500 + var_y
    var_z = var_y - b / 200

    var_y = tf.where(var_y > 0.008856, tf.pow(var_y, 3),
                     (var_y - 16 / 116) / 7.787)

    var_x = tf.where(var_x > 0.008856, tf.pow(var_x, 3),
                     (var_x - 16 / 116) / 7.787)

    var_z = tf.where(var_z > 0.008856, tf.pow(var_z, 3),
                     (var_z - 16 / 116) / 7.787)

    refx = 95.047
    refy = 100.00
    ref_z = 108.883

    x = var_x * refx
    y = var_y * refy
    z = var_z * ref_z
    return tf.stack([x, y, z])


def xyz_to_rgb():
    pass

def rgb_xyz():
    pass


def rgb_to_lab():
    pass
