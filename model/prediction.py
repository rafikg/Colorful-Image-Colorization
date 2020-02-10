import tensorflow as tf
import skimage.io as io
import skimage.transform as transform
from model import ImageColorizedModel
from utils import lab_to_rgb, rgb_to_lab

loss_object = tf.keras.losses.CategoricalCrossentropy(name='Cross_entropy')
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.optimizers.SGD(learning_rate=1e-3)

model = ImageColorizedModel(loss_object=loss_object, optimizer=optimizer,
                            train_loss=train_loss, is_training=True)

# fit the model on any input to just create the weights
x = tf.random.uniform(shape=(1, 256, 256, 1))
y = tf.random.uniform(shape=(1, 64, 64, 1))
dataset = [(x, y)]
model.fit(dataset=dataset, epochs=1)
model.load_weights('../model_weights/colorful_model_weoghs.h5')

sample = io.imread('../images/index.jpeg')
sample = transform.resize(sample, (256, 256))
image = tf.convert_to_tensor(sample)

lab = rgb_to_lab(image)

l_channel = tf.expand_dims(lab[:, :, 0], axis=-1)
l_channel = tf.expand_dims(l_channel, axis=0)
rgb = model.predict(l_channel)
