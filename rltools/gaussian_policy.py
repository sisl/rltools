import numpy as np
import tensorflow as tf

import nn
import tfutil
from policy import StochasticPolicy
from distributions import Gaussian


class GaussianMLPPolicy(StochasticPolicy):
    def __init__(self, obsfeat_space, action_space,
                 hidden_spec, enable_obsnorm, min_stdev, init_logstdev, tblog, varscope_name):
        self.hidden_spec = hidden_spec
        self.min_stdev = min_stdev
        self.init_logstev = init_logstdev
        self._dist = Gaussian(action_space.n)
        super(GaussianMLPPolicy, self).__init__(obsfeat_space, action_space,
                                                action_space.n*2, # Mean and diagonal stdev
                                                enable_obsnorm, tblog, varscope_name)

    @property
    def distribution(self):
        return self._dist

    def _make_actiondist_ops(self, obsfeat_B_Df):
        with tf.variable_scope('hidden'):
            net = nn.FeedforwardNet(obsfeat_B_Df, self.obsfeat_space.shape, self.hidden_spec)
        with tf.variable_scope('out'):
            mean_layer = nn.AffineLayer(net.output, net.output_shape, (self.action_space.n,), initializer=tf.zeros_initializer)

        means_B_Da = mean_layer.output

        # logstdev params
        logstdevs_1_Da = tf.get_variable('logstdevs_1_Da', shape=(1, self.action_space.n), initializer=tf.constant_initializer(self.init_logstdev))
        stdevs_1_Da = self.min_stdev + tf.exp(logstdevs_1_Da) # Required for stability of kl computations
        stdevs_B_Da = tf.ones_like(means_B_Da)*stdevs_1_Da

        actiondist_B_Pa = tf.concat(1, [means_B_Da, stdevs_B_Da])
        return actiondist_B_Pa

    def _extract_actiondist_params(self, actiondist_B_Pa):
        means_B_Da = actiondist_B_Pa[:, :self.action_space.n]
        stdevs_B_Da = actiondist_B_Pa[:, self.action_space.n:]
        return means_B_Da, stdevs_B_Da

    def _make_actiondist_logprobs_ops(self, actiondist_B_Pa, input_actions_B_Da):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return self.distribution.log_density(means_B_Da, stdevs_B_Da, input_actions_B_Da)

    def _make_actiondist_kl_ops(self, proposal_actiondist_B_Pa, actiondist_B_Pa):
        return self.distribution.kl_expr(*map(self._extract_actiondist_params, [proposal_actiondist_B_Pa, actiondist_B_Pa]))

    def _sample_from_actiondist(self, actiondist_B_Pa, deterministic=False):
        means_B_Da, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        if deterministic:
            return means_B_Da
        stdnormal_B_Da = np.random.randn(actiondist_B_Pa.shape[0], self.action_space.n)
        assert stdnormal_B_Da.shape == means_B_Da.shape == stdevs_B_Da.shape
        return (stdnormal_B_Da*stdevs_B_Da) + means_B_Da

    def _compute_actiondist_entropy(self, actiondist_B_Pa):
        _, stdevs_B_Da = self._extract_actiondist_params(actiondist_B_Pa)
        return self.distribution.entropy(stdevs_B_Da)
