from typing import Tuple
import tensorflow as tf
import copy


def crop_or_pad_image(x: tf.Tensor, y: tf.Tensor, in_h: int, in_w: int,
                      out_h: int, out_w: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
        Random crop or pad x and y image with zero
        Parameters
        ----------
        x: tf.Tensor
        y: tf.Tensor
        in_h: int
        in_w: int
        out_h:int
        out_w:int
        Returns
        -------
        x, y: Tuple(tf.Tensor, tf.Tensor)
        """
    x_shape = tf.shape(x)
    last_dim_x = x_shape[-1]
    concat = tf.concat([x, y], axis=-1)
    concat_crop = tf.image.resize_with_crop_or_pad(
        image=concat,
        target_height=in_h,
        target_width=in_w
    )

    x_crop = concat_crop[:, :, :last_dim_x]
    y_crop = concat_crop[:, :, last_dim_x:]
    y_crop = tf.image.resize(y_crop,
                             size=(out_h,
                                   out_w))
    return x_crop, y_crop


def flip(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Flip horizontally x image
    Parameters
    ----------
    x : tf.tensor

    Returns
    -------
    Flipped images Tuple(tf.Tensor, tf.Tensor)
    """
    x = tf.image.flip_left_right(x)
    return x, x
