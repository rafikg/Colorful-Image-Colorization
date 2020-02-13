import numpy as np
import os
import tensorflow as tf

PATH_FILE = os.path.join(os.path.split(__file__)[0],
                         'quantazation/pts_in_hull.npy')
CENTERS = tf.constant(np.load(PATH_FILE))
T = 0.38
EPSILON = 1e-6
NUM_CLASSES_Q = 313
