from __future__ import absolute_import, print_function

import gym
import numpy as np
import tensorflow as tf

from .context import rltools

import rltools.algos.policyopt
import rltools.log
import rltools.util
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.categorical import CategoricalMLPPolicy
from rltools.samplers.serial import SimpleSampler

SIMPLE_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''


def run(baseline_cls):
    env = gym.make('CartPole-v0')
    policy = CategoricalMLPPolicy(env.observation_space, env.action_space, hidden_spec=SIMPLE_ARCH,
                                  enable_obsnorm=True, tblog='/tmp/madrl', varscope_name='policy')
    if baseline_cls == MLPBaseline:
        baseline = baseline_cls(env.observation_space, SIMPLE_ARCH, True, True, max_kl=0.01,
                                damping=0.1, time_scale=1. / 200, varscope_name='baseline')
    elif baseline_cls == ZeroBaseline:
        baseline = baseline_cls(env.observation_space)
    elif baseline_cls == LinearFeatureBaseline:
        baseline = baseline_cls(env.observation_space, True)
    else:
        raise NotImplementedError()

    sampler_cls = SimpleSampler
    sampler_args = dict(max_traj_len=200, batch_size=4, min_batch_size=4, max_batch_size=64,
                        batch_rate=40, adaptive=False)

    step_func = rltools.algos.policyopt.TRPO(max_kl=0.01)

    popt = rltools.algos.policyopt.SamplingPolicyOptimizer(env=env, policy=policy,
                                                           baseline=baseline, step_func=step_func,
                                                           discount=0.99, gae_lambda=1.,
                                                           sampler_cls=sampler_cls,
                                                           sampler_args=sampler_args, n_iter=1)

    log_f = rltools.log.TrainingLog(None, [], False)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f, 1)


def test_baselines():
    baselines = [ZeroBaseline, LinearFeatureBaseline, MLPBaseline]
    for bcls in baselines:
        tf.reset_default_graph()
        run(bcls)
        assert True
