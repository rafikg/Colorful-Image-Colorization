import tensorflow as tf
import datetime
import logging
from tensorflow.keras.callbacks import (ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard, EarlyStopping)

from data import ColorfulDataset
from model import ImageColorizedModel
from config import NUM_CLASSES_Q

logging.basicConfig(format="%(levelname)s:%(name)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Define distribution strategy
strategy = tf.distribute.MirroredStrategy()

# logger info device

logger.info(
    'Training will run on {} GPUs:'.format(strategy.num_replicas_in_sync))

# Select the batch size per replica
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# Select epochs
EPOCHS = 10

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=3e-5)

logger.info('Start training')

# Define the dataset object
dataset_train_obj = ColorfulDataset(
    path="../dataset/train_data",
    img_ext="*.jpg",
    batch_size=BATCH_SIZE,
    n_workers=8)
dataset_valid_obj = ColorfulDataset(
    path="../dataset/valid_data",
    img_ext="*.jpg",
    batch_size=BATCH_SIZE,
    n_workers=8,
    is_validation=True,
    is_training=False)

train_dataset = dataset_train_obj.tf_data
valid_dataset = dataset_valid_obj.tf_data

# Define the model inside a strategy.scope
with strategy.scope():
    # create the model
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
reduce_lr = ReduceLROnPlateau(monitor='valid_loss', patience=10)
early_stop = EarlyStopping('valid_loss', patience=10)

callbacks = [tensorboard, model_checkpoint, reduce_lr, early_stop]

# Start training the model
model.fit(x=train_dataset, validation_data=valid_dataset,
          epochs=EPOCHS, callbacks=callbacks)
