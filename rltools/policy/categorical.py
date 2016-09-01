from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from rltools import nn, tfutil
from rltools.distributions import Categorical, RecurrentCategorical
from rltools.policy.stochastic import StochasticPolicy
from rltools.util import EzPickle


class CategoricalMLPPolicy(StochasticPolicy, EzPickle):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, tblog,
                 varscope_name):
        EzPickle.__init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, tblog,
                          varscope_name)
        self.hidden_spec = hidden_spec
        self._dist = Categorical(action_space.n)
        super(CategoricalMLPPolicy, self).__init__(observation_space, action_space, action_space.n,
                                                   enable_obsnorm, tblog, varscope_name)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.observation_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            out_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.n,),
                                       Winitializer=tf.zeros_initializer,
                                       binitializer=None)  # TODO action_space

        scores_B_Pa = out_layer.output
        actiondist_B_Pa = scores_B_Pa - tfutil.logsumexp(scores_B_Pa, axis=1)
        return actiondist_B_Pa

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        return self.distribution.log_density_expr(actiondist_B_Pa, input_actions_B_Da[:, 0])

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return self.distribution.kl_expr(proposal_actiondist_B_Pa, actiondist_B_Pa)

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        probs_B_A = np.exp(actiondist_B_Pa)
        assert probs_B_A.shape[1] == self.action_space.n
        if deterministic:
            return np.argmax(probs_B_A, axis=1)[:, None]
        return self.distribution.sample(probs_B_A)[:, None]

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        return self.distribution.entropy(np.exp(actiondist_B_Pa))


class CategoricalGRUPolicy(StochasticPolicy):

    def __init__(self, observation_space, action_space, hidden_spec, enable_obsnorm, tblog,
                 varscope_name, state_include_action=True):
        self.hidden_spec = hidden_spec
        self.state_include_action = state_include_action
        self._dist = RecurrentCategorical(action_space.n)
        super(CategoricalGRUPolicy, self).__init__(observation_space, action_space, action_space.n,
                                                   enable_obsnorm, tblog, varscope_name)

    @property
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist
