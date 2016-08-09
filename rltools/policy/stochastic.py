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
            # Action distribution for current policy
            self._obsfeat_B_Df = tf.placeholder(
                tf.float32, list((batch_size,) + self.obsfeat_space.shape),
                name='obsfeat_B_Df')  # Df = feature dimensions FIXME shape
            with tf.variable_scope('obsnorm'):
                self.obsnorm = (nn.Standardizer if enable_obsnorm else
                                nn.NoOpStandardizer)(self.obsfeat_space.shape[0])
            self._normalized_obsfeat_B_Df = self.obsnorm.standardize_expr(self._obsfeat_B_Df)
            self._actiondist_B_Pa = self._make_actiondist_ops(
                self._normalized_obsfeat_B_Df)  # Pa = action distribution params
            self._input_action_B_Da = tf.placeholder(
                action_type, [batch_size, action_dim],
                name='input_actions_B_Da')  # Action dims FIXME type
            self._logprobs_B = self._make_actiondist_logprobs_ops(self._actiondist_B_Pa,
                                                                  self._input_action_B_Da)

            # proposal distribution from old policy
            self._proposal_actiondist_B_Pa = tf.placeholder(tf.float32,
                                                            [batch_size, num_actiondist_params],
                                                            name='proposal_actiondist_B_Pa')
            self._proposal_logprobs_B = self._make_actiondist_logprobs_ops(
                self._proposal_actiondist_B_Pa, self._input_action_B_Da)

            # Advantage
            self._advantage_B = tf.placeholder(tf.float32, [batch_size], name='advantage_B')

            # Plain pg objective (REINFORCE)
            impweight_B = tf.exp(self._logprobs_B - self._proposal_logprobs_B)
            self._reinfobj = tf.reduce_mean(impweight_B * self._advantage_B)  # Surrogate loss

            # KL
            self._kl_coeff = tf.placeholder(tf.float32, name='kl_cost_coeff')
            kl_B = self._make_actiondist_kl_ops(self._proposal_actiondist_B_Pa,
                                                self._actiondist_B_Pa)
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

            self._ngstep = optim.make_ngstep_func(
                self, compute_obj_kl=self.compute_reinfobj_kl,
                compute_obj_kl_with_grad=self.compute_reinfobj_kl_with_grad,
                compute_hvp_helper=self.compute_klgrad)

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

    def _compute_internal_normalized_obsfeat(self, sess, obsfeat_B_Df):
        return sess.run(self._normalized_obsfeat_B_Df, {self._obsfeat_B_Df: obsfeat_B_Df})

    def compute_action_dist_params(self, sess, obsfeat_B_Df):
        """Actually evaluate action distribution params"""
        return sess.run(self._actiondist_B_Pa, {self._obsfeat_B_Df: obsfeat_B_Df})

    def sample_actions(self, sess, obsfeat_B_Df, deterministic=False):
        """Sample actions conditioned on observations
        (Also returns the params)
        """
        actiondist_B_Pa = self.compute_action_dist_params(sess, obsfeat_B_Df)
        return self._sample_from_actiondist(actiondist_B_Pa, deterministic), actiondist_B_Pa

    def deterministic_action(self, sess, obsfeat_B_Df):
        """Return the argmax of distribution
        """
        actiondist_B_Pa = self.compute_action_dist_params(sess, obsfeat_B_Df)
        a = self._sample_from_actiondist(actiondist_B_Pa, deterministic=True)
        return a[0, 0]

    def compute_action_logprobs(self, sess, obsfeat_B_Df, actions_B_Da):
        return sess.run(self._logprobs_B, {self._obsfeat_B_Df: obsfeat_B_Df,
                                           self._input_action_B_Da: actions_B_Da})

    def compute_kl_cost(self, sess, obsfeat_B_Df, proposal_actiondist_B_Pa):
        return sess.run(self._kl, {self._obsfeat_B_Df: obsfeat_B_Df,
                                   self._proposal_actiondist_B_Pa: proposal_actiondist_B_Pa})

    def compute_reinfobj_kl(self, sess, obsfeat_B_Df, input_action_B_Da, proposal_actiondist_B_Pa,
                            advantage_B):
        return sess.run([self._reinfobj, self._kl],
                        {self._obsfeat_B_Df: obsfeat_B_Df,
                         self._input_action_B_Da: input_action_B_Da,
                         self._proposal_actiondist_B_Pa: proposal_actiondist_B_Pa,
                         self._advantage_B: advantage_B})

    def compute_reinfobj_kl_with_grad(self, sess, obsfeat_B_Df, input_action_B_Da,
                                      proposal_actiondist_B_Pa, advantage_B):
        return sess.run([self._reinfobj, self._kl, self._reinfobj_grad_P],
                        {self._obsfeat_B_Df: obsfeat_B_Df,
                         self._input_action_B_Da: input_action_B_Da,
                         self._proposal_actiondist_B_Pa: proposal_actiondist_B_Pa,
                         self._advantage_B: advantage_B})

    def compute_klgrad(self, sess, obsfeat_B_Df, input_action_B_Da, proposal_actiondist_B_Pa,
                       advantage_B):
        return sess.run(self._kl_grad_P, {self._obsfeat_B_Df: obsfeat_B_Df,
                                          self._proposal_actiondist_B_Pa: proposal_actiondist_B_Pa
                                         })  # TODO check if we need more

    # TODO penobj computes

    def set_params(self, sess, params_P):
        sess.run(self._assign_params, {self._flatparams_P: params_P})

    def get_params(self, sess):
        params_P = sess.run(self._curr_params_P)
        assert params_P.shape == (self._num_params,)
        return params_P

    def get_state(self, sess):
        state_PA = sess.run(self._curr_all_params_PA)
        assert state_PA.shape == (self._num_all_params,)
        return state_PA

    def set_state(self, sess, state_PA):
        sess.run(self._assign_all_params, {self._flatallparams_PA: state_PA})

    @contextmanager
    def try_params(self, sess, params_D):
        orig_params_D = self.get_params(sess)
        self.set_params(sess, params_D)
        yield  # Do what you gotta do
        self.set_params(sess, orig_params_D)

    def take_desent_step(self, sess, grad_P, learning_rate):
        sess.run(self._take_descent_step, {self._flatparams_P: grad_P,
                                           self._learning_rate: learning_rate})
