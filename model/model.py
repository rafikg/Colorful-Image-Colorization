import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.layers import (Conv2D, UpSampling2D,
                                     BatchNormalization)
from tensorflow.keras.models import Model
from utils import ImageReader

# Destroys the current TF graph
tf.keras.backend.clear_session()


class ImageColorizedModel(Model):
    """
    Encapsulates the architecture of image colorized network
    """

    def __init__(self, loss_object, optimizer, train_loss,
                 num_classes=313, name='ColorfulImage', is_training=True):
        super().__init__(name=name)
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.is_training = is_training

        # block1
        self.conv1_1 = Conv2D(filters=64, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv1_2 = Conv2D(filters=64, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn1 = BatchNormalization(trainable=self.is_training)

        # block2
        self.conv2_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv2_2 = Conv2D(filters=128, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn2 = BatchNormalization(trainable=self.is_training)

        # block3
        self.conv3_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv3_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv3_3 = Conv2D(filters=256, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn3 = BatchNormalization(trainable=self.is_training)

        # block4
        self.conv4_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv4_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv4_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn4 = BatchNormalization(trainable=self.is_training)

        # block5
        self.conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv5_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.bn5 = BatchNormalization(trainable=self.is_training)

        # block6
        self.conv6_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv6_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv6_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.bn6 = BatchNormalization(trainable=self.is_training)

        # block7
        self.conv7_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv7_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv7_3 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn7 = BatchNormalization(trainable=self.is_training)

        # block 8
        self.up_samp = UpSampling2D(size=(2, 2))
        self.conv8_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv8_2 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv8_3 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn8 = BatchNormalization(trainable=self.is_training)
        self.final_output = Conv2D(num_classes, (1, 1), activation='softmax',
                             padding='same',
                             name='pred')

    def call(self, x):
        # block1
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.bn1(x)

        # block2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.bn2(x)

        # block3
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.bn3(x)

        # block4
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.bn4(x)

        # block5
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.bn5(x)

        # block6
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.bn6(x)

        # block7
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        x = self.conv7_3(x)
        x = self.bn7(x)

        # block 8
        x = self.up_samp(x)
        x = self.conv8_1(x)
        x = self.conv8_2(x)
        x = self.conv8_3(x)
        x = self.bn8(x)
        x = self.final_output(x)
        return x

    @tf.function
    def train_one_step(self, x, y):
        """
        train the model for one step
        Parameters
        ----------
        x : tf.Tensor
        y: tf.Tensor

        Returns
        -------

        """
        with tf.GradientTape as tape:
            output = self.__call__(x)
            loss = self.loss_object(y, output)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.train_loss(loss)

    def fit(self, dataset, epochs, train_log_dir):
        """
        fit the model on the dataset
        Parameters
        ----------
        dataset: tf.data.Dataset
            dataset object
        epochs: int
            number of epochs
        train_log_dir: str

        Returns
        -------
        self: trained model
        """
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        for epoch in range(epochs):
            for step, (l, ab) in enumerate(dataset):
                if epoch == 0 and step == 0:
                    tf.summary.trace_on(graph=True, profiler=True)
                self.train_one_step(x=l, y=ab)
                if epoch == 0 and step == 0:
                    with train_summary_writer.as_default():
                        tf.summary.trace_export(
                            name="my_func_trace",
                            step=0,
                            profiler_outdir=train_log_dir)
                    tf.summary.trace_off()
            template = 'Epoch {}, Loss: {}'
            print(template.format(epoch + 1, self.train_loss.result()))
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)

            # reset the metric for the next epoch
            self.train_loss.reset_states()

    def predict(self, x: tf.Tensor):
        pass

    def evaluate(self, x: tf.Tensor, y: tf.Tensor):
        pass


if __name__ == '__main__':
    BATCH_SIZE = 8
    EPOCHS = 10

    loss_object = tf.keras.losses.MeanSquaredError(name='mse')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    optimizer = tf.optimizers.Adam(learning_rate=1e-3, name='ADAM')

    # Set up summary writers to write the summaries to disk
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    img_reader = ImageReader(
        img_path='../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages',
        ext="*.jpg", height=256, width=256, is_training=True,
        batch_size=BATCH_SIZE,
        n_workers=8, epochs=EPOCHS)
    dataset = img_reader.dataset

    model = ImageColorizedModel(loss_object=loss_object, optimizer=optimizer,
                                train_loss=train_loss, is_training=True)
    X = np.random.rand(1, 256, 256, 1)
    y = model(X)

    model.fit(dataset=dataset, epochs=EPOCHS, train_log_dir=train_log_dir)
