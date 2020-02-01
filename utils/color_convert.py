import tensorflow as tf


# all functions are implemented based on
# <http://www.easyrgb.com/en/math.php#text8>


def lab_to_xyz(l: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from LAB color space to XYZ color space
    Parameters
    ----------
    l: tf.Tensor
    a: tf.Tensor
    b : tf.Tensor

    Returns
    -------
    tf.Tensor : LAB image
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
    xyz_image = tf.stack([x, y, z])
    return xyz_image


def xyz_to_rgb(x: tf.Tensor, y: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from XYZ color space to RGB color space
    Parameters
    ----------
    x: tf.Tensor
    y: Tf.Tensor
    z: tf.Tensor

    Returns
    -------
    tf.Tensor: RGB image
    """
    var_x = x / 100
    var_y = y / 100
    var_z = z / 100

    var_r = var_x * 3.2406 + var_y * -1.5372 + var_z * -0.4986
    var_g = var_x * -0.9689 + var_y * 1.8758 + var_z * 0.0415
    var_b = var_x * 0.0557 + var_y * -0.2040 + var_z * 1.0570

    var_r = tf.where(var_r > 0.0031308,
                     1.055 * tf.pow(var_r, (1 / 2.4)) - 0.055,
                     12.92 * var_r)
    var_g = tf.where(var_g > 0.0031308,
                     1.055 * tf.pow(var_g, (1 / 2.4)) - 0.055,
                     12.92 * var_g)
    var_b = tf.where(var_b > 0.0031308,
                     1.055 * tf.pow(var_b, (1 / 2.4)) - 0.055,
                     12.92 * var_b)
    r = var_r * 255
    g = var_g * 255
    b = var_b * 255
    rgb_image = tf.stack([r, g, b])
    return rgb_image


def rgb_xyz(r: tf.Tensor, g: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from RGB color space to XYZ color space
    Parameters
    ----------
    r: tf.Tensor
    g: tf.Tensor
    b: tf.Tensor

    Returns
    -------
    tf.tensor: XYZ image
    """
    var_r = r / 255
    var_g = g / 255
    var_b = b / 255

    var_r = tf.where(var_r > 0.04045, tf.pow((var_r + 0.055) / 1.055, 2.4),
                     var_r / 12.92)
    var_g = tf.where(var_g > 0.04045, tf.pow((var_g + 0.055) / 1.055, 2.4),
                     var_g / 12.92)
    var_b = tf.where(var_b > 0.04045, tf.pow((var_b + 0.055) / 1.055, 2.4),
                     var_b / 12.92)
    var_r = var_r * 100
    var_g = var_g * 100
    var_b = var_b * 100

    x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    image_xyz = tf.stack([x, y, z])
    return image_xyz


def rgb_to_lab(r: tf.Tensor, g: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from RGB color space to LAB color space
    RGB -> XYZ -> LAB
    Parameters
    ----------
    r: tf.Tensor
    g: tf.Tensor
    b: tf.Tensor

    Returns
    -------
    tf.tensor: LAB image
    """
    xyz = rgb_xyz(r, g, b)
    x, y, z = tf.unstack(xyz, axis=-1)
    lab = xyz


def lab_to_rgb(l: tf.Tensor, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Convert an image from LAB color space to RGB color space
    LAB -> XYZ -> RGB
    Parameters
    ----------
    l: tf.Tensor
    a: tf.Tensor
    b : tf.Tensor

    Returns
    -------
    tf.Tensor: RGB image
    """
    xyz = lab_to_xyz(l, a, b)
    x, y, z = tf.unstack(xyz)
    rgb = xyz_to_rgb(x, y, z)
    return rgb
