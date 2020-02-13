import tensorflow as tf
import datetime
import logging
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard, EarlyStopping)

from data import ColorfulDataset
from model import ImageColorizedModel
from config import NUM_CLASSES_Q

logger = logging.getLogger(__name__)

# Select batch size
BATCH_SIZE = 32
# Select epochs
EPOCHS = 2

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=3e-5)

logger.info('Start training')

# Define the dataset object
dataset_obj = ColorfulDataset(
    path="../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/",
    img_ext="*.jpg",
    n_workers=8)

dataset = dataset_obj.tf_data

# Define the model
model = ImageColorizedModel(num_classes=NUM_CLASSES_Q,
                            is_training=True)

# Compile the model
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Define Callbacks list
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = '../logs/' + current_time + '/train'
model_tag = '../model_weights/model.{epoch:02d}'
tensorboard = TensorBoard(log_dir=train_log_dir, histogram_freq=0,
                          write_graph=True, write_images=True,
                          update_freq=10)
model_checkpoint = ModelCheckpoint(filepath=model_tag, monitor='loss',
                                   verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', patience=10)
early_stop = EarlyStopping('loss', patience=10)

callbacks = [tensorboard, model_checkpoint, reduce_lr, early_stop]

# Start training the model
model.fit(x=dataset, epochs=EPOCHS, callbacks=callbacks)
