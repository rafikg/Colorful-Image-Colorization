import tensorflow as tf
from utils import rgb_to_lab, lab_to_rgb
import skimage.io as io
import skimage.transform as transform
import skimage.color as color
import matplotlib.pyplot as plt
from model import ImageColorizedModel
from config import EPSILON, T, NUM_CLASSES_Q, CENTERS

model = ImageColorizedModel(num_classes=313, is_training=False)

model.load_weights('../model_weights/model.01')

# prepare the input
sample_rgb = io.imread('../images/index.jpeg').astype('float64')

sample_rgb = transform.resize(sample_rgb, output_shape=(256, 256))
# convert from rgb to lab
lab_img = rgb_to_lab(sample_rgb)

l_channel = lab_img[:, :, 0]
a_channel = lab_img[:, :, 1]
b_channel = lab_img[:, :, 1]

# convert to grayscale
sample_gray = color.rgb2gray(sample_rgb)

# Normalize image
sample_gray = sample_gray / 255

# convert image to tensor
sample_gray = tf.expand_dims(
    tf.expand_dims(tf.convert_to_tensor(sample_gray), axis=-1), axis=0)

output = model(sample_gray)
output = tf.cast(output, tf.float64)
_, w, h, _ = output.shape
output = tf.reshape(output, (w * h, NUM_CLASSES_Q))
output = tf.exp(tf.math.log(output + EPSILON) / T)
output = output / tf.expand_dims(tf.reduce_sum(output, 1), axis=-1)

q_a = tf.reshape(CENTERS[:, 0], (1, NUM_CLASSES_Q))
q_b = tf.reshape(CENTERS[:, 1], (1, NUM_CLASSES_Q))

output_a = tf.reduce_sum(output * q_a, 1)
output_a = tf.reshape(output_a, (w, h, 1))

output_a = tf.cast(tf.image.resize(output_a, (256, 256)), tf.float64)

output_b = tf.reduce_sum(output * q_b, 1)
output_b = tf.reshape(output_b, (w, h, 1))

output_b = tf.cast(tf.image.resize(output_b, (256, 256)), tf.float64)

output_a = output_a + 128
output_b = output_b + 128
output_lab = tf.stack([l_channel, output_a[:, :, 0], output_b[:, :, 0]],
                      axis=-1)

rgb_output = lab_to_rgb(output_lab)
plt.imshow(rgb_output)
plt.show()
