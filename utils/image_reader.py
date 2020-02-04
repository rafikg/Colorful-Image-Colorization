import tensorflow as tf
import glob
import os
from utils import rgb_to_lab, get_lightness, get_ab
from typing import List, Callable, Tuple


class ImageReader(object):
    """
    Image Reader which read images from the disk and enques them into a
    tensorflow queue using tf.data.Dataset
    """

    def __init__(self, img_path: str, ext: str = "*.jpg", height: int = 224,
                 width: int = 224, is_training: bool = True,
                 batch_size: int = 16,
                 n_workers: int = 8,
                 epochs: int = 100):
        self.imgs_list = self.read_images_list(img_path, ext)
        self.height = height
        self.width = width
        self.is_training = is_training
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.dataset = self.create_dataset()
        self.epochs = epochs

    def read_images_list(self, imgs_path: str, ext: str = '*.jpg') -> List:
        """
        Read images list
        Parameters
        ----------
        imgs_path: str

        Returns
        -------
        list: image files list
        """
        img_list = []
        for file in glob.glob(os.path.join(imgs_path, ext)):
            img_list.append(file)
        return img_list

    def parse_image(self, image_filename) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Read the image file name
        Returns
        -------
        tuple(tf.Tensor, tf.Tensor): rgb raw data
        """
        img_contents = tf.io.read_file(image_filename)

        # decode the image
        img = tf.image.decode_jpeg(img_contents, channels=3)

        return img, img

    def create_dataset(self) -> Callable:
        """
        Create tf.data.Dataset
        Returns
        -------
        tf.data.Dataset object
        """

        dataset = tf.data.Dataset.from_tensor_slices(
            (self.imgs_list, self.imgs_list))
        dataset = dataset.map(lambda x, y: self.parse_image(x),
                              num_parallel_calls=self.n_workers)
        dataset = dataset.map(lambda x, y: self.transform_image_label(x),
                              num_parallel_calls=self.n_workers)
        dataset = dataset.map(lambda x, y: self.crop_or_pad_image(x, y),
                              num_parallel_calls=self.n_workers)
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=1024)
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.repeat(1)
            dataset = dataset.prefetch(buffer_size=1024)
        else:
            dataset = dataset.batch(1)

        return dataset

    def transform_image_label(self, x: tf.Tensor) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """
        Transform image from RGB to lab and return
        L and ab channels
        Parameters
        ----------
        x: tf.tensor
            input image
        y: tf.tensor

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
        """
        lab = rgb_to_lab(x)
        l_channel = get_lightness(lab)
        ab_channels = get_ab(lab)
        return l_channel, ab_channels

    def crop_or_pad_image(self, x: tf.Tensor, y: tf.Tensor) -> Tuple[
        tf.Tensor, tf.Tensor]:
        """
        Random crop or pad x and y image with zero
        Parameters
        ----------
        x: tf.Tensor
        y: tf.Tensor

        Returns
        -------
        x, y: Tuple(tf.Tensor, tf.Tensor)
        """
        x_shape = tf.shape(x)
        last_dim_x = x_shape[-1]
        concat = tf.concat([x, y], axis=-1)
        concat_crop = tf.image.resize_with_crop_or_pad(
            image=concat,
            target_height=self.height,
            target_width=self.width
        )

        x_crop = concat_crop[:, :, :last_dim_x]
        y_crop = concat_crop[:, :, last_dim_x:]

        return x_crop, y_crop


if __name__ == '__main__':
    Img_reader_gen = ImageReader(
        img_path='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages',
        ext='*.jpg', height=224, width=224, is_training=True)

    for x, y in Img_reader_gen.dataset:
        print(x.shape, y.shape)
