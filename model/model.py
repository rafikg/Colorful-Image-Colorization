import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, UpSampling2D,
                                     BatchNormalization)
from tensorflow.keras.models import Model
from quantazation import T, CENTERS, EPSILON, NUM_CLASSES_Q
from utils import lab_to_rgb
# Destroys the current TF graph
tf.keras.backend.clear_session()


class ImageColorizedModel(Model):
    """
    Encapsulates the architecture of image colorized network
    """

    def __init__(self, loss_object, optimizer, train_loss,
                 num_classes=NUM_CLASSES_Q, name='ColorfulImage',
                 is_training=True):
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
                                   padding='same', name='pred')

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
        with tf.GradientTape() as tape:
            output = self.__call__(x)
            loss = self.loss_object(y, output)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.train_loss(loss)

    def fit(self, dataset, epochs, train_log_dir=None):
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
        if train_log_dir:
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        template_step = 'Epoch: {}, step: {}, Loss: {}'
        template_epoch = 'Epoch: {}, Loss: {}'
        for epoch in range(epochs):
            for step, (l, ab) in enumerate(dataset):
                self.train_one_step(x=l, y=ab)
                print(template_step.format(epoch, step,
                                           self.train_loss.result()))
            print(template_epoch.format(epoch, self.train_loss.result()))
            if train_log_dir:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(),
                                      step=epoch)
            # reset the metric for the next epoch
            self.train_loss.reset_states()

    def predict(self, x: tf.Tensor):
        """
        run inference pipeline on input x
        Parameters
        ----------
        x: tf.Tensor
            L channel of lab color space

        Returns
        -------

        """
        _, img_w, img_h, _ = x.shape
        class_prob = self.__call__(x)
        q_a = tf.reshape(tf.cast(CENTERS[:, 0], 'float32'), (1, NUM_CLASSES_Q))
        q_b = tf.reshape(tf.cast(CENTERS[:, 1], 'float32'), (1, NUM_CLASSES_Q))
        _, w, h, ch = class_prob.shape
        class_prob = tf.reshape(class_prob, (w * h, NUM_CLASSES_Q))
        class_prob = tf.exp(tf.math.log(class_prob + EPSILON) / T)
        class_prob = class_prob / tf.expand_dims(
            tf.reduce_sum(class_prob, axis=1), axis=1)
        # Reweight the class_prob
        im_a_channel = tf.reshape(tf.reduce_sum(class_prob * q_a, axis=1),
                                  (w, h, 1))
        im_b_channel = tf.reshape(tf.reduce_sum(class_prob * q_b, axis=1),
                                  (w, h, 1))
        # resize a and b channel
        im_a_channel = tf.image.resize(im_a_channel, (w * 4, h * 4))
        im_b_channel = tf.image.resize(im_b_channel, (w * 4, h * 4))

        lab = tf.stack(
            [x[0, :, :, 0], im_a_channel[:, :, 0], im_b_channel[:, :, 0]],
            axis=-1)
        rgb = lab_to_rgb(lab)
        return rgb
