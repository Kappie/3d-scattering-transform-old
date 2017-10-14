import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from plot_slices import plot2d, plot3d

from wavelets import filter_bank

x = y = z = 10
js = [0, 1]
J = 1
L = 2

with tf.Session() as sess:
    filters = filter_bank(x, y, z, js, J, L)
    signal = filters['psi'][0][0]
    signal_spatial = tf.ifft3d(signal)
    result = sess.run(signal_spatial)

plot3d(np.real(result))
