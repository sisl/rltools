import numpy as np
import tensorflow as tf

import tfutil
import util

TINY = 1e-10


class Distribution(object):

    @property
    def dim(self):
        raise NotImplementedError()

    def kl(self, old, new):
        raise NotImplementedError()

    def log_density(self, dist_params, x):
        raise NotImplementedError()

    def entropy(self, logprobs_N_K):
        raise NotImplementedError()

    def sample(self, logprobs_N_K):
        raise NotImplementedError()

    def kl_expr(self, logprobs1, logprobs2):
        raise NotImplementedError()

    def log_density_expr(self, dist_params, x):
        raise NotImplementedError()


class Categorical(Distribution):

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def log_density(self, dist_params_B_A, x_B_A):
        return util.lookup_last_idx(dist_params_B_A, x_B_A)

    def entropy(self, probs_N_K):
        tmp = -probs_N_K * np.log(probs_N_K + TINY)
        tmp[~np.isfinite(tmp)] = 0
        return tmp.sum(axis=1)

    def sample(self, probs_N_K):
        """Sample from N categorical distributions, each over K outcomes"""
        N, K = probs_N_K.shape
        return np.array([np.random.choice(K, p=probs_N_K[i, :]) for i in xrange(N)])

    def kl_expr(self, logprobs1_B_A, logprobs2_B_A, name=None):
        """KL divergence between categorical distributions, specified as log probabilities"""
        with tf.op_scope([logprobs1_B_A, logprobs2_B_A], name, 'categorical_kl') as scope:
            kl_B = tf.reduce_sum(
                tf.exp(logprobs1_B_A) * (logprobs1_B_A - logprobs2_B_A), 1, name=scope)
        return kl_B

    def log_density_expr(self, dist_params_B_A, x_B_A):
        """Log density from categorical distribution params"""
        return tfutil.lookup_last_idx(dist_params_B_A, x_B_A)

class Gaussian(Distribution):

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def entropy(self, stdevs_B_A):
        d = stdevs_B_A.shape[1]
        return .5 * d * (1. + np.log(2. * np.pi)) + np.log(stdevs_B_A).sum(axis=1)

    def kl_expr(self, means1_B_A_stdevs1_B_A, means2_B_A_stdevs2_B_A, name=None):
        """KL divergence wbw diagonal covariant gaussians"""
        means1_B_A, stdevs1_B_A = means1_B_A_stdevs1_B_A
        means2_B_A, stdevs2_B_A = means2_B_A_stdevs2_B_A
        with tf.op_scope([means1_B_A, stdevs1_B_A, means2_B_A, stdevs2_B_A], name,
                         'gaussian_kl') as scope:
            D = tf.shape(means1_B_A)[1]
            kl_B = tf.mul(.5, (tf.reduce_sum(
                tf.square(stdevs1_B_A / stdevs2_B_A), 1) + tf.reduce_sum(
                    tf.square((means2_B_A - means1_B_A) / stdevs2_B_A), 1) + 2. * (tf.reduce_sum(
                        tf.log(stdevs2_B_A), 1) - tf.reduce_sum(
                            tf.log(stdevs1_B_A), 1)) - tf.to_float(D)), name=scope)
        return kl_B

    def log_density_expr(self, means_B_A, stdevs_B_A, x_B_A, name=None):
        """Log density of diagonal gauss"""
        with tf.op_scope([means_B_A, stdevs_B_A, x_B_A], name, 'gauss_log_density') as scope:
            D = tf.shape(means_B_A)[1]
            lognormconsts_B = -.5 * tf.to_float(D) * np.log(2. * np.pi) + 2. * tf.reduce_sum(
                tf.log(stdevs_B_A), 1)  # log norm consts
            logprobs_B = tf.add(-.5 * tf.reduce_sum(
                tf.square((x_B_A - means_B_A) / stdevs_B_A), 1), lognormconsts_B, name=scope)
        return logprobs_B
