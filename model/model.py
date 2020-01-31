import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import(Conv2D, UpSampling2D,
    BatchNormalization)
from tensorflow.keras.models import Model


class ImageColorizedModel(Model):
    """
    Encapsulates the architecture of image colorized network
    """

    def __init__(self, loss_object, optimizer, train_loss,
                 name='ColorfulImage'):
        super().__init__(name=name)
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.train_loss = train_loss

        # block1
        self.conv1_1 = Conv2D(filters=64, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv1_2 = Conv2D(filters=64, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn1 = BatchNormalization()

        # block2
        self.conv2_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv2_2 = Conv2D(filters=128, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn2 = BatchNormalization()

        # block3
        self.conv3_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv3_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv3_3 = Conv2D(filters=256, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same')
        self.bn3 = BatchNormalization()

        # block4
        self.conv4_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv4_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv4_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn4 = BatchNormalization()

        # block5
        self.conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv5_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.bn5 = BatchNormalization()

        # block6
        self.conv6_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv6_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.conv6_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same')
        self.bn6 = BatchNormalization()

        # block7
        self.conv7_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv7_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv7_3 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn7 = BatchNormalization()

        # block 8
        self.up_samp = UpSampling2D(size=(2, 2))
        self.conv8_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv8_2 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.conv8_3 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same')
        self.bn8 = BatchNormalization()

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
        return x

    @tf.function
    def train_one_step(self, x):
        with tf.GradientTape as tape:
            output = self.__call__(x)
            loss = self.loss_object(x, output)
            loss += self.losses + loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.train_loss(loss)

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    model = ImageColorizedModel(loss_object=None, optimizer=None,
                                train_loss=None)

    x = np.random.rand(1, 224, 224, 1)
    y = model(x)
    print(y.shape)
