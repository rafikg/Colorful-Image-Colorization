import tensorflow as tf
import datetime
from utils import ImageReader
from model import ImageColorizedModel


BATCH_SIZE = 16
EPOCHS = 10

loss_object = tf.keras.losses.CategoricalCrossentropy(name='Cross_entropy')
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.optimizers.SGD(learning_rate=1e-3)

# Set up summary writers to write the summaries to disk
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '../logs/gradient_tape/' + current_time + '/train'

img_reader = ImageReader(
    img_path='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages',
    ext="*.jpg", height=256, width=256, is_training=True,
    batch_size=BATCH_SIZE,
    n_workers=12, epochs=EPOCHS)
dataset = img_reader.dataset

model = ImageColorizedModel(loss_object=loss_object, optimizer=optimizer,
                            train_loss=train_loss, is_training=True)

model.fit(dataset=dataset, epochs=EPOCHS, train_log_dir=train_log_dir)
model.save_weights('colorful_model_weoghs.h5')
