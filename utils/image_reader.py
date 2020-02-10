import tensorflow as tf
import glob
import os
import time
from utils import rgb_to_lab, get_lightness, get_ab
from typing import List, Callable, Tuple
from quantazation import CENTERS, NUM_CLASSES_Q

tf.keras.backend.clear_session()


class ImageReader(object):
    """
    Image Reader which read images from the disk and enques them into a
    tensorflow queue using tf.data.Dataset
    """

    def __init__(self, img_path: str, ext: str = "*.jpg", height: int = 256,
                 width: int = 256, is_training: bool = True,
                 batch_size: int = 16,
                 n_workers: int = 8,
                 epochs: int = 100,
                 down_scale: int = 4):
        self.imgs_list = self.read_images_list(img_path, ext)
        self.height = height
        self.width = width
        self.is_training = is_training
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.epochs = epochs
        self.target_height = height // down_scale
        self.target_width = width // down_scale
        self.dataset = self.create_dataset()

    def __len__(self):
        return len(self.imgs_list)

    def read_images_list(self, imgs_path: str, ext: str = '*.jpg') -> List:
        """
        Read images list
        Parameters
        ----------
        imgs_path: str
            images path
        ext: str
            image extension

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
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.map(lambda x, y: self.transform_image_label(x),
                              num_parallel_calls=self.n_workers)
        dataset = dataset.map(lambda x, y: self.crop_or_pad_image(x, y),
                              num_parallel_calls=self.n_workers)
        dataset = dataset.map(lambda x, y: self.quantize(x, y))
        if self.is_training:
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=1)
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
        y_crop = tf.image.resize(y_crop,
                                 size=(self.target_height,
                                       self.target_width))
        return x_crop, y_crop

    def quantize(self, x: tf.Tensor, y: tf.Tensor):
        """
        Quantize ab_channels
        Parameters
        ----------
        x: l_channel
        y: ab_channels

        Returns
        -------
        x: l_channel
        y: quantized ab_channels

        """
        h, w, _ = y.shape
        y = tf.reshape(y, (-1, 2))
        distances, indices = self.__knn(centers=CENTERS, sample=y, k=5)
        # smooth the distances with a gaussian kernel
        gauss_sigma = 5
        # # TODO inspect this operation (check nan values)
        # distances = tf.exp(-distances ** 2 / (2 * gauss_sigma ** 2))
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
    def __knn(self, centers: tf.Tensor, sample: tf.Tensor, k):
        """
        Get k Nearest neighbour of the sample
        Parameters
        ----------
        centers: tf.Tensor
        sample: tf.Tensor
        k: int
            k nearest neighbors

        Returns
        -------

        """
        centers = tf.cast(centers, 'float32')
        # X^2
        centers_sqr = tf.expand_dims(tf.reduce_sum(tf.pow(centers, 2), axis=1),
                                     axis=0)
        # Y^2
        sample_sqr = tf.expand_dims(tf.reduce_sum(tf.pow(sample, 2), axis=1),
                                    axis=1)

        # Repeat X^2 n times vertically
        centers_sqrR = tf.repeat(centers_sqr, self.target_height ** 2, axis=0)

        # Repeat Y^2 m times
        sample_sqrR = tf.repeat(sample_sqr, NUM_CLASSES_Q, axis=1)

        # X*Y
        cross = tf.matmul(sample, tf.transpose(centers))

        # X^2 + Y^2 - 2*X*Y
        distances = tf.sqrt(centers_sqrR + sample_sqrR - 2 * cross)

        # Return top_k distances and indices
        dist, indx = tf.math.top_k(-1 * distances, k=k, sorted=True)
        return -1 * dist, indx


if __name__ == '__main__':
    Img_reader_gen = ImageReader(
        img_path='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages',
        ext='*.jpg', height=256, width=256, is_training=True, n_workers=12)
    start = time.time()
    for x, y in Img_reader_gen.dataset:
        print(x.shape, " ", y.shape)
    end = time.time()
    print("time {}".format(end - start))
