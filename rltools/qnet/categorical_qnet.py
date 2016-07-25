import numpy as np
import tensorflow as tf

from gym import spaces
from rltools.nn import Model, FeedforwardNet, AffineLayer
from rltools import tfutil


class CategoricalQFunction(Model):

    def __init__(self, obsfeat_space, action_space, hidden_spec, learning_rate, varscope_name,
                 primary_q_func=None, dueling=False):
        self.obsfeat_space = obsfeat_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.hidden_spec = hidden_spec
        self.dueling = dueling
        self.min_delta = -1
        self.max_delta = 1

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            if isinstance(self.action_space, spaces.Discrete):
                action_type = tf.float32
                if hasattr(self.action_space, 'ndim'):
                    self.action_dim = self.action_space.ndim
                else:
                    self.action_dim = 1
            else:
                raise NotImplementedError()

            self._obsfeat_B_Df = tf.placeholder(tf.float32,
                                                list((batch_size,) + self.obsfeat_space.shape),
                                                name='obsfeat_B_Df')
            # TODO Normalization
            self._action_B_Da = tf.placeholder(action_type, [batch_size, self.action_dim])

            #
            self._qvals_B = self._make_qval_op(self._obsfeat_B_Df, self._action_B_Da)

            # Loss for Q learning
            self._qtargets_B = tf.placeholder(tf.float32, [batch_size])
            self._delta = self._qvals_B - self._qtargets_B
            self._loss = tf.reduce_mean(
                tf.square(tf.clip_by_value(self._delta, self.min_delta, self.max_delta)),
                name='loss')

            with tf.variable_scope('optimizer'):
                self._optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self._loss)

            # Reading params
            self._param_vars = self.get_trainable_variables()
            self._num_params = self.get_num_params()
            self._curr_params_P = tfutil.flatcat(self._param_vars)  # Flatten the params and concat

            # Writing params
            self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
            # For updating vars directly
            self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self._param_vars)

            assert isinstance(primary_q_func, CategoricalQFunction) or primary_q_func is None
            self.primary_q_func = primary_q_func

    def _make_qval_op(self, obsfeat_B_Df, action_B_Da):
        if self.dueling:
            # FIXME
            with tf.variable_scope('hidden'):
                input_B_Di = tf.concat(1, [obsfeat_B_Df, action_B_Da])
                net = FeedforwardNet(input_B_Di, (self.obsfeat_space.shape[0] + self.action_dim,),
                                     self.hidden_spec)
            with tf.variable_scope('value_hidden'):
                value_net = FeedforwardNet(net.output, net.output_shape, self.value_hidden_spec)
            with tf.variable_scope('adv_hidden'):
                adv_net = FeedforwardNet(net.output, net.output_shape, self.adv_hidden_spec)
            with tf.variable_scope('value_out'):
                value_out_layer = AffineLayer(value_net.output, value_net.output_shape, (1,),
                                              initializer=tf.zeros_initializer)
            with tf.variable_scope('adv_out'):
                adv_out_layer = AffineLayer(adv_net.output, adv_net.output.shape,
                                            (self.action_dim,), initializer=tf.zeros_initializer)

            qvals_B_D = value_out_layer.output + (adv_out_layer.output - tf.reduce_mean(
                adv_out_layer.output, reduction_indices=1, keep_dims=True))
            return qvals_B_D[:, 0]
        else:
            with tf.variable_scope('hidden'):
                input_B_Di = tf.concat(1, [obsfeat_B_Df, action_B_Da])
                net = FeedforwardNet(input_B_Di, (self.obsfeat_space.shape[0] + self.action_dim,),
                                     self.hidden_spec)
            with tf.variable_scope('out'):
                out_layer = AffineLayer(net.output, net.output_shape, (1,),
                                        initializer=tf.zeros_initializer)
            assert out_layer.output_shape == (1,)
            qvals_B = out_layer.output[:, 0]
        return qvals_B

    def compute_qvals(self, sess, obsfeat_B_Df, action_B_Da):
        return sess.run(self._qvals_B, {self._obsfeat_B_Df: obsfeat_B_Df,
                                        self._action_B_Da: action_B_Da})

    def compute_qactions(self, sess, obsfeat_B_Df):
        if len(obsfeat_B_Df.shape) > 1:
            batch_size = len(obsfeat_B_Df)
        else:
            batch_size = 1
            obsfeat_B_Df = np.expand_dims(obsfeat_B_Df, 0)
        acts = [np.ones((batch_size, 1)) * idx for idx in range(self.action_space.n)]
        action_NB_Da = np.concatenate(acts, axis=0)
        assert action_NB_Da.shape == (self.action_space.n * batch_size, 1)
        obsfeat_NB_Df = np.tile(obsfeat_B_Df, (self.action_space.n, 1))
        assert obsfeat_NB_Df.shape == (self.action_space.n * batch_size, obsfeat_B_Df.shape[-1])
        qvals_B_N = self.compute_qvals(sess, obsfeat_NB_Df,
                                       action_NB_Da).reshape(self.action_space.n, -1).T
        assert qvals_B_N.shape == (len(obsfeat_B_Df), self.action_space.n)
        actions_B_Da = np.argmax(qvals_B_N, 1)[:, None]
        assert actions_B_Da.shape == (batch_size, 1)
        return actions_B_Da

    def opt_step(self, sess, obsfeat_B_Df, action_B_Da, qtargets_B):
        loss, _ = sess.run([self._loss, self._optimizer], {self._obsfeat_B_Df: obsfeat_B_Df,
                                                           self._action_B_Da: action_B_Da,
                                                           self._qtargets_B: qtargets_B})
        return loss

    def eval_loss(self, sess, obsfeat_B_Df, action_B_Da, qtargets_B):
        return sess.run(self._loss, {self._obsfeat_B_Df: obsfeat_B_Df,
                                     self._action_B_Da: action_B_Da,
                                     self._qtargets_B: qtargets_B})

    def copy_params_from_primary(self, sess):
        assert self.primary_q_func is not None
        params_P = self.primary_q_func.get_params(sess)
        self.set_params(sess, params_P)

    def interp_params_with_primary(self, sess, alpha):
        assert self.primary_q_func is not None
        params_P = self.get_params(sess)
        other_params_P = self.primary_q_func.get_params(sess)
        new_params_P = alpha * other_params_P + (1. - alpha) * params_P
        self.set_params(sess, new_params_P)

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P
