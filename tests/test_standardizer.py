from __future__ import print_function
from __future__ import absolute_import

from .context import rltools
import numpy as np

import tensorflow as tf


def test_standardizer():
    D = 10
    s = rltools.nn.Standardizer(D, eps=0)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        x_N_D = np.random.randn(200, D)
        s.update(sess, x_N_D)

        x2_N_D = np.random.rand(300, D)
        s.update(sess, x2_N_D)

        allx = np.concatenate([x_N_D, x2_N_D], axis=0)

        assert np.allclose(s.get_mean(sess)[0, :], allx.mean(axis=0))
        assert np.allclose(s.get_stdev(sess)[0, :], allx.std(axis=0))
