import numpy as np
import os
import tensorflow as tf
import sklearn.neighbors as knn
PATH_FILE = os.path.join(os.path.split(__file__)[0], 'pts_in_hull.npy')
CENTERS = tf.constant(np.load(PATH_FILE))
T = 0.38
EPSILON = 1e-6
KNN_NEIGHB = knn.NearestNeighbors(n_neighbors=5, metric='euclidean').fit(CENTERS)
NUM_CLASSES_Q = 313
