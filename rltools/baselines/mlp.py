from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from rltools import nn, optim, tfutil
from rltools.baselines import Baseline


class MLPBaseline(Baseline, nn.Model):
    """ Multi Layer Perceptron baseline

    Optimized using natural gradients.
    """

    def __init__(self, obsfeat_space, hidden_spec, enable_obsnorm, enable_vnorm, max_kl, damping,
                 varscope_name, subsample_hvp_frac=.1, grad_stop_tol=1e-6, time_scale=1.):
        self.obsfeat_space = obsfeat_space
        self.hidden_spec = hidden_spec
        self.enable_obsnorm = enable_obsnorm
        self.enable_vnorm = enable_vnorm
        self.max_kl = max_kl
        self.damping = damping
        self.subsample_hvp_frac = subsample_hvp_frac
        self.grad_stop_tol = grad_stop_tol
        self.time_scale = time_scale

        with tf.variable_scope(varscope_name) as self.varscope:
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.obsfeat_space.shape)
            with tf.variable_scope('vnorm'):
                self.vnorm = (nn.Standardizer if enable_vnorm else nn.NoOpStandardizer)(1)

            batch_size = None
            self.obsfeat_B_Df = tf.placeholder(tf.float32,
                                               list((batch_size,) + self.obsfeat_space.shape),
                                               name='obsfeat_B_Df')  # FIXME shape
            self.t_B_1 = tf.placeholder(tf.float32, [batch_size, 1], name='t_B')
            scaled_t_B_1 = self.t_B_1 * self.time_scale
            self.val_B = self._make_val_op(self.obsfeat_B_Df, scaled_t_B_1)

        # Only code above has trainable vars
        self.param_vars = self.get_variables(trainable=True)
        self._num_params = self.get_num_params(trainable=True)
        self._curr_params_P = tfutil.flatcat(self.param_vars)

        # Squared loss for fitting the value function
        self.target_val_B = tf.placeholder(tf.float32, [batch_size], name='target_val_B')
        self.obj = -tf.reduce_mean(tf.square(self.val_B - self.target_val_B))
        self.objgrad_P = tfutil.flatcat(tf.gradients(self.obj, self.param_vars))

        # KL divergence (as Gaussian) and its gradient
        self.old_val_B = tf.placeholder(tf.float32, [batch_size], name='old_val_B')
        self.kl = tf.reduce_mean(tf.square(self.old_val_B - self.val_B))

        # KL gradient
        self.kl_grad_P = tfutil.flatcat(tf.gradients(self.kl, self.param_vars))

        # Writing params
        self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
        self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self.param_vars)

        self._ngstep = optim.make_ngstep_func(
            self, compute_obj_kl=self.compute_obj_kl,
            compute_obj_kl_with_grad=self.compute_obj_kl_with_grad,
            compute_hvp_helper=self.compute_klgrad)

    def _make_val_op(self, obsfeat_B_Df, scaled_t_B_1):
        if len(obsfeat_B_Df.get_shape()) > 2:
            with tf.variable_scope('in'):
                flatshape = (np.prod(obsfeat_B_Df.get_shape().as_list()[1:]),)
                flatobsfeat_B_Df = nn.ReshapeLayer(obsfeat_B_Df, flatshape).output
        else:
            flatshape = self.obsfeat_space.shape
            flatobsfeat_B_Df = obsfeat_B_Df
        net_input = tf.concat(1, [flatobsfeat_B_Df, scaled_t_B_1])
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(net_input, (flatshape[0] + 1,), self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (1,),
                                       initializer=tf.zeros_initializer)
            assert out_layer.output_shape == (1,)
        return out_layer.output[:, 0]

    def compute_obj_kl(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.obj, self.kl], {self.obsfeat_B_Df: obsfeat_B_Df,
                                              self.t_B_1: t_B[:, None],
                                              self.target_val_B: target_val_B,
                                              self.old_val_B: old_val_B})

    def compute_obj_kl_with_grad(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.obj, self.kl, self.objgrad_P], {self.obsfeat_B_Df: obsfeat_B_Df,
                                                              self.t_B_1: t_B[:, None],
                                                              self.target_val_B: target_val_B,
                                                              self.old_val_B: old_val_B})

    def compute_klgrad(self, sess, obsfeat_B_Df, t_B, target_val_B, old_val_B):
        return sess.run([self.kl_grad_P], {self.obsfeat_B_Df: obsfeat_B_Df,
                                           self.t_B_1: t_B[:, None],
                                           self.target_val_B: target_val_B,
                                           self.old_val_B: old_val_B})[0]

    def update_obsnorm(self, sess, obs_B_Do):
        """Update norms using moving avg"""
        self.obsnorm.update(sess, obs_B_Do)

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    @contextmanager
    def try_params(self, sess, params_P):
        orig_params_P = self.get_params(sess)
        self.set_params(sess, params_P)
        yield
        self.set_params(sess, orig_params_P)

    def _predict_raw(self, sess, obsfeat_B_Df, t_B):
        return sess.run(self.val_B, {self.obsfeat_B_Df: obsfeat_B_Df, self.t_B_1: t_B[:, None]})

    def fit(self, sess, trajs, qval_B):
        obs_B_Do = trajs.obs.stacked
        t_B = trajs.time.stacked

        # Update norm
        self.obsnorm.update(sess, obs_B_Do)
        self.vnorm.update(sess, qval_B[:, None])

        # Take step
        sobs_B_Do = self.obsnorm.standardize(sess, obs_B_Do)
        sqval_B = self.vnorm.standardize(sess, qval_B[:, None])[:, 0]
        feed = (sobs_B_Do, t_B, sqval_B, self._predict_raw(sess, sobs_B_Do, t_B))
        step_info = self._ngstep(sess, feed, max_kl=self.max_kl, damping=self.damping,
                                 subsample_hvp_frac=self.subsample_hvp_frac,
                                 grad_stop_tol=self.grad_stop_tol)
        return [
            ('vf_dl', step_info.obj1 - step_info.obj0, float),  # Improvement in objective
            ('vf_kl', step_info.kl1, float),  # kl cost
            ('vf_gnorm', step_info.gnorm, float),  # gradient norm
            ('vf_bt', step_info.bt, int),  # backtracking steps
        ]

    def predict(self, sess, trajs):
        obs_B_Do = trajs.obs.stacked
        t_B = trajs.time.stacked
        sobs_B_Do = self.obsnorm.standardize(sess, obs_B_Do)
        pred_B = self.vnorm.unstandardize(sess, self._predict_raw(sess, sobs_B_Do,
                                                                  t_B)[:, None])[:, 0]
        assert pred_B.shape == t_B.shape
        return pred_B
