from __future__ import absolute_import, print_function

from collections import namedtuple
from contextlib import contextmanager

import tensorflow as tf
from gym import spaces

from rltools import nn, optim, tfutil, util
from rltools.policy import Policy


class StochasticPolicy(Policy):

    def __init__(self, obsfeat_space, action_space, num_actiondist_params, enable_obsnorm, tblog,
                 varscope_name):
        super(StochasticPolicy, self).__init__(obsfeat_space, action_space)

        with tf.variable_scope(varscope_name) as self.varscope:
            batch_size = None
            if isinstance(self.action_space, spaces.Discrete):
                action_type = tf.int32
                if hasattr(self.action_space, 'ndim'):
                    action_dim = self.action_space.ndim
                else:
                    action_dim = 1
            elif isinstance(self.action_space, spaces.Box):
                action_type = tf.float32
                action_dim = self.action_space.shape[0]
            else:
                raise NotImplementedError()

            if self.recurrent:
                obsfeat_shape = list((batch_size,
                                      None,) + self.obsfeat_space.shape)
            else:
                obsfeat_shape = list((batch_size,) + self.obsfeat_space.shape)

            # Action distribution for current policy
            self._obsfeat_B_Df = tf.placeholder(
                tf.float32, obsfeat_shape,
                name='obsfeat_B_Df')  # Df = feature dimensions FIXME shape
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.obsfeat_space.shape)
            self._normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(self._obsfeat_B_Df)

            if self.recurrent:
                action_shape = list((batch_size, None, action_dim))
            else:
                action_shape = list((batch_size, action_dim))

            if self.recurrent:
                self._actiondist_B_Pa, self._hidden_vec = self._make_actiondist_ops(
                    self._normalized_obsfeat_B_Df)  # Pa = action distribution params
            else:
                self._actiondist_B_Pa = self._make_actiondist_ops(
                    self._normalized_obsfeat_B_Df)  # Pa = action distribution params

            self._input_action_B_Da = tf.placeholder(
                action_type, action_shape, name='input_actions_B_Da')  # Action dims FIXME type

            if self.recurrent:
                self._logprobs_B = self._make_actiondist_logprobs_ops(self._actiondist_B_Pa,
                                                                      self._hidden_vec,
                                                                      self._input_action_B_Da)
            else:
                self._logprobs_B = self._make_actiondist_logprobs_ops(self._actiondist_B_Pa,
                                                                      self._input_action_B_Da)

            if self.recurrent:
                # TODO
                # proposal distribution from old policy
                self._proposal_actiondist_B_Pa = tf.placeholder(tf.float32,
                                                                [batch_size, num_actiondist_params],
                                                                name='proposal_actiondist_B_Pa')
                self._proposal_logprobs_B = self._make_actiondist_logprobs_ops(
                    self._proposal_actiondist_B_Pa, self._input_action_B_Da)
            else:
                # proposal distribution from old policy
                self._proposal_actiondist_B_Pa = tf.placeholder(tf.float32,
                                                                [batch_size, num_actiondist_params],
                                                                name='proposal_actiondist_B_Pa')
                self._proposal_logprobs_B = self._make_actiondist_logprobs_ops(
                    self._proposal_actiondist_B_Pa, self._input_action_B_Da)

            # Advantage
            if self.recurrent:
                self._advantage_B = tf.placeholder(tf.float32, [batch_size, None],
                                                   name='advantage_B')
            else:
                self._advantage_B = tf.placeholder(tf.float32, [batch_size], name='advantage_B')

            if self.recurrent:
                self._valid = tf.placeholder(tf.float32, shape=[None, None], name="valid")
            else:
                self._valid = None
            # Plain pg objective (REINFORCE)
            impweight_B = tf.exp(self._logprobs_B - self._proposal_logprobs_B)
            if self.recurrent:
                self._reinfobj = tf.reduce_sum(impweight_B * self._advantage_B *
                                               self._valid) / tf.reduce_sum(self._valid)
            else:
                self._reinfobj = tf.reduce_mean(impweight_B * self._advantage_B)  # Surrogate loss

            # KL
            self._kl_coeff = tf.placeholder(tf.float32, name='kl_cost_coeff')
            kl_B = self._make_actiondist_kl_ops(self._proposal_actiondist_B_Pa,
                                                self._actiondist_B_Pa)
            if self.recurrent:
                self._kl = tf.reduce_sum(kl_B * self._valid) / tf.reduce_sum(self._valid)
            else:
                self._kl = tf.reduce_mean(kl_B, 0)  # Minimize kl divergence

            # KL Penalty objective for PPO
            self._penobj = self._reinfobj - self._kl_coeff * self._kl

            # All trainable vars done (only _make_* methods)

            # Reading params
            self._param_vars = self.get_variables(trainable=True)
            self._num_params = self.get_num_params(trainable=True)
            self._curr_params_P = tfutil.flatcat(self._param_vars)  # Flatten the params and concat

            self._all_param_vars = self.get_variables()
            self._num_all_params = self.get_num_params()
            self._curr_all_params_PA = tfutil.flatcat(self._all_param_vars)

            # Gradients of objective
            self._reinfobj_grad_P = tfutil.flatcat(tf.gradients(self._reinfobj, self._param_vars))
            self._penobj_grad_P = tfutil.flatcat(tf.gradients(self._penobj, self._param_vars))

            # KL gradient for TRPO
            self._kl_grad_P = tfutil.flatcat(tf.gradients(self._kl, self._param_vars))

            # Writing params
            self._flatparams_P = tf.placeholder(tf.float32, [self._num_params], name='flatparams_P')
            # For updating vars directly, e.g. for PPO
            self._assign_params = tfutil.unflatten_into_vars(self._flatparams_P, self._param_vars)

            self._flatallparams_PA = tf.placeholder(tf.float32, [self._num_all_params],
                                                    name='flatallparams_PA')
            self._assign_all_params = tfutil.unflatten_into_vars(self._flatallparams_PA,
                                                                 self._all_param_vars)

            # Treats placeholder self._flatparams_p as gradient for descent
            with tf.variable_scope('optimizer'):
                self._learning_rate = tf.placeholder(tf.float32, name='learning_rate')
                vargrads = tfutil.unflatten_into_tensors(
                    self._flatparams_P, [v.get_shape().as_list() for v in self._param_vars])
                self._take_descent_step = tf.train.AdamOptimizer(
                    learning_rate=self._learning_rate).apply_gradients(
                        util.safezip(vargrads, self._param_vars))

            self._tbwriter = tf.train.SummaryWriter(tblog, graph=tf.get_default_graph())

        # Functions
        ins = [self._obsfeat_B_Df,
               self._input_action_B_Da,
               self._proposal_actiondist_B_Pa,
               self._advantage_B,]
        if self.recurrent:
            ins.append(self._valid)

        self._compute_internal_normalized_obsfeat = tfutil.function(self._obsfeat_B_Df,
                                                                    self._normalized_obsfeat_B_Df)
        self.compute_action_logprobs = tfutil.function(
            [self._obsfeat_B_Df, self._input_action_B_Da], self._logprobs_B)
        self.compute_action_dist_params = tfutil.function([self._obsfeat_B_Df],
                                                          self._actiondist_B_Pa)

        self.compute_kl_cost = tfutil.function(ins, self._kl)
        self.compute_klgrad = tfutil.function(ins, self._kl_grad_P)
        self.compute_reinfobj_kl = tfutil.function(ins, [self._reinfobj, self._kl])
        self.compute_reinfobj_kl_with_grad = tfutil.function(
            ins, [self._reinfobj, self._kl, self._kl_grad_P])

        self._ngstep = optim.make_ngstep_func(
            self, compute_obj_kl=self.compute_reinfobj_kl,
            compute_obj_kl_with_grad=self.compute_reinfobj_kl_with_grad,
            compute_hvp_helper=self.compute_klgrad)

        self.set_params = tfutil.function([self._flatparams_P], [], [self._assign_params])
        self.get_params = tfutil.function([], self._curr_params_P)
        self.get_state = tfutil.function([], self._curr_all_params_PA)
        self.set_state = tfutil.function([self._flatallparams_PA], [], [self._assign_all_params])

    @property
    def distribution(self):
        raise NotImplementedError()

    def update_obsnorm(self, sess, obs_B_Do):
        """Update norms using moving avg"""
        self.obsnorm.update(sess, obs_B_Do)

    def _make_actiondist_ops(self, obsfeat_B_Df):
        """Ops to compute action distribution parameters

        For Gaussian, these would be mean and std
        For categorical, these would be log probabilities
        """
        raise NotImplementedError()

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        raise NotImplementedError()

    def _make_actiondist_kl_ops(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        raise NotImplementedError()

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        raise NotImplementedError()

    def sample_actions(self, obsfeat_B_Df, deterministic=False):
        """Sample actions conditioned on observations
        (Also returns the params)
        """
        actiondist_B_Pa = self.compute_action_dist_params(obsfeat_B_Df)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    def deterministic_action(self, obsfeat_B_Df):
        """Return the argmax of distribution
        """
        actiondist_B_Pa = self.compute_action_dist_params(obsfeat_B_Df)
        a = self._sample_from_actiondist(actiondist_B_Pa, deterministic=True)
        return a[0, 0]

    @contextmanager
    def try_params(self, params_D):
        orig_params_D = self.get_params()
        self.set_params(params_D)
        yield  # Do what you gotta do
        self.set_params(orig_params_D)

    def take_descent_step(self, sess, grad_P, learning_rate):
        sess.run(self._take_descent_step, {self._flatparams_P: grad_P,
                                           self._learning_rate: learning_rate})
