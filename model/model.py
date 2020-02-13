from typing import Callable

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, UpSampling2D,
                                     BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

# Destroys the current TF graph
tf.keras.backend.clear_session()


class ImageColorizedModel(Model):
    """
    Encapsulates the architecture of image colorized network
    """

    def __init__(self,
                 num_classes: int,
                 is_training: bool = True,
                 l2_reg: Callable = l2(1e-3),
                 name: str = 'ColorfulImage'):
        super().__init__(name=name)
        self.is_training = is_training

        # block1
        self.conv1_1 = Conv2D(filters=64, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg)
        self.conv1_2 = Conv2D(filters=64, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn1 = BatchNormalization(trainable=self.is_training)

        # block2
        self.conv2_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv2_2 = Conv2D(filters=128, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn2 = BatchNormalization(trainable=self.is_training)

        # block3
        self.conv3_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv3_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv3_3 = Conv2D(filters=256, kernel_size=3, strides=2,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn3 = BatchNormalization(trainable=self.is_training)

        # block4
        self.conv4_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv4_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv4_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn4 = BatchNormalization(trainable=self.is_training)

        # block5
        self.conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv5_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn5 = BatchNormalization(trainable=self.is_training)

        # block6
        self.conv6_1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv6_2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv6_3 = Conv2D(filters=512, kernel_size=3, strides=1,
                              dilation_rate=2, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn6 = BatchNormalization(trainable=self.is_training)

        # block7
        self.conv7_1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv7_2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv7_3 = Conv2D(filters=256, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn7 = BatchNormalization(trainable=self.is_training)

        # block 8
        self.up_samp = UpSampling2D(size=(2, 2))
        self.conv8_1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv8_2 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.conv8_3 = Conv2D(filters=128, kernel_size=3, strides=1,
                              dilation_rate=1, padding='same',
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2_reg
                              )
        self.bn8 = BatchNormalization(trainable=self.is_training)
        self.final_output = Conv2D(num_classes, (1, 1), activation='softmax',
                                   padding='same', name='final_output')

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
