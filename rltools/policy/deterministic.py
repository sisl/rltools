from collections import namedtuple
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from gym import spaces

from rltools import nn, optim, tfutil, util
from rltools.policy import Policy


class DeterministicPolicy(Policy):

    def __init__(self, obsfeat_space, action_space, enable_obsnorm, tblog, varscope_name):
        super(DeterministicPolicy, self).__init__(obsfeat_space, action_space)

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            if isinstance(action_space, spaces.Discrete):
                action_type = tf.int32
                action_dim = 1
            elif isinstance(action_space, spaces.Box):
                action_type = tf.float32
                action_dim = self.action_space.shape[0]
            else:
                raise NotImplementedError()

            self._obsfeat_B_Df = tf.placeholder(
                tf.float32, list((batch_size,) + self.obsfeat_space.shape), nmae='obsfeat_B_Df')
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.obsfeat_space.shape[0])
            self._normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(self._obsfeat_B_Df)

            self._action_B_Da = self._make_action_ops(self._normalized_obsfeat_B_Df)

            # All jacobians
            self._param_vars = self.get_variables(trainable=True)
            self._num_params = self.get_num_params(trainable=True)
