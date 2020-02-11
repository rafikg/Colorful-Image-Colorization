from typing import List, Tuple, Callable

from pathlib import Path
from . import get_l_and_ab_channels, flip, crop_or_pad_image, quantize
import glob
import os
import tensorflow as tf


class ColorfulDataset(object):
    """
    Read images from the disk and enques them into a tensorflow queue using
    tf.data.Dataset
    """

    def __init__(self, path: str, img_ext: str,
                 in_height: int = 256,
                 in_width: int = 256,
                 down_scale: int = 4,
                 batch_size: int = 16,
                 n_workers: int = 12,
                 is_cached: bool = False,
                 is_flip: bool = True,
                 is_parallel: bool = False,
                 is_training: bool = True,
                 is_validation: bool = False,
                 is_shuffle: bool = True,
                 name='Pascal2012'):

        self.path = Path(path)
        self.img_ext = img_ext
        self.in_height = in_height
        self.in_width = in_width
        self.down_scale = int(down_scale)
        self.batch_size = batch_size
        self.n_workers = n_workers

        self.is_cached = is_cached
        self.is_flip = is_flip
        self.is_parallel = is_parallel
        self.is_training = is_training
        self.is_validation = is_validation
        self.is_shuffle = is_shuffle

        self.name = name

        self._out_height = in_height // down_scale
        self._out_width = in_width // down_scale

        if not Path.exists(self.path):
            raise IOError(f"{self.path}' is invalid path")

        self.imgs_list = self.load_files_list()
        self.tf_data = self.create_dataset()

    def load_files_list(self) -> List:
        """
        Get all files paths inside the dataset path
        Returns
        -------

        """
        files_list = []
        for file in glob.glob(os.path.join(self.path, self.img_ext)):
            files_list.append(file)
        return files_list

    def image_decoding(self, x: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Decode the image and return the raw data
        Returns
        -------

        """
        img_contents = tf.io.read_file(x)

        # decode the image
        img = tf.image.decode_jpeg(img_contents, channels=3)
        img = img / 255
        return img, img

    def create_dataset(self) -> Callable:
        """
        Create a tf.data.Dataset
        Returns
        -------
        tf.data.Dataset object
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.imgs_list,
                                                      self.imgs_list))

        # load images
        dataset = dataset.map(lambda x, y: self.image_decoding(x),
                              num_parallel_calls=self.n_workers)
        # cached images
        if self.is_cached:
            dataset = dataset.cache()

        # shuffle images
        if self.is_shuffle:
            dataset = dataset.shuffle(buffer_size=128)

        # flip images horizontally
        if self.is_flip:
            if tf.random.uniform(shape=(1,)) > 0.5:
                dataset = dataset.map(lambda x, y: flip(x),
                                      num_parallel_calls=self.n_workers)

        # convert rgb to lab
        dataset = dataset.map(lambda x, y: get_l_and_ab_channels(x),
                              num_parallel_calls=self.n_workers)

        # crop or pad images
        dataset = dataset.map(
            lambda x, y: crop_or_pad_image(x=x, y=y,
                                           in_h=self.in_height,
                                           in_w=self.in_width,
                                           out_h=self._out_height,
                                           out_w=self._out_width),
            num_parallel_calls=self.n_workers)
        # quantize image
        dataset = dataset.map(lambda x, y: quantize(x=x, y=y),
                              num_parallel_calls=self.n_workers)

        if self.is_training:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=1)
        else:
            dataset = dataset.batch(1)

        return dataset
