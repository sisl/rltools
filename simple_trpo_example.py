#!/usr/bin/env python
#
# File: main.py
#
# Created: Wednesday, July  6 2016 by rejuvyesh <mail@rejuvyesh.com>
#
# Example: ./simple_trpo_example.py --env CartPole-v0 --n_iter 60 --save_freq 30 --log cart_pole.h5

from __future__ import absolute_import, print_function

import argparse
import json

import tensorflow as tf

import gym
import rltools.algos.policyopt
import rltools.log
import rltools.util
from rltools.samplers.serial import SimpleSampler, ImportanceWeightedSampler, DecSampler
from rltools.samplers.parallel import ParallelSampler
from rltools.baselines.linear import LinearFeatureBaseline
from rltools.baselines.mlp import MLPBaseline
from rltools.baselines.zero import ZeroBaseline
from rltools.policy.categorical import CategoricalMLPPolicy

SIMPLE_POLICY_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''
SIMPLE_VAL_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "tanh"}
    ]
    '''


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='CartPole-v0')  # gym environment

    parser.add_argument('--discount', type=float, default=0.95)
    parser.add_argument('--gae_lambda', type=float, default=0.99)

    parser.add_argument('--sampler', type=str, default='simple')
    parser.add_argument('--sampler_workers', type=int, default=4)

    parser.add_argument('--n_iter', type=int, default=250)  # trpo iterations
    parser.add_argument('--max_traj_len', type=int, default=200)  # max length of a trajectory (ts)
    parser.add_argument('--n_timesteps', type=int, default=8000)  # number of traj in an iteration

    parser.add_argument('--policy_hidden_spec', type=str,
                        default=SIMPLE_POLICY_ARCH)  # policy net architecture
    parser.add_argument('--baseline_hidden_spec', type=str,
                        default=SIMPLE_VAL_ARCH)  # baseline value net architecture

    # TRPO params
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--vf_max_kl', type=float, default=0.01)
    parser.add_argument('--vf_cg_damping', type=float, default=0.01)

    parser.add_argument('--save_freq', type=int, default=20)  # how often checkpoints are made
    parser.add_argument('--log', type=str, required=False)  # log file
    parser.add_argument('--tblog', type=str, default='/tmp/madrl_tb')  # tensorboard log dir

    args = parser.parse_args()

    # init environment
    env = gym.make(args.env)

    # init policy network
    policy = CategoricalMLPPolicy(env.observation_space, env.action_space,
                                  hidden_spec=args.policy_hidden_spec, enable_obsnorm=True,
                                  varscope_name='pursuit_catmlp_policy')

    # init baseline
    baseline = MLPBaseline(env.observation_space, args.baseline_hidden_spec, True, True,
                           max_kl=args.vf_max_kl, damping=args.vf_cg_damping,
                           time_scale=1. / args.max_traj_len, varscope_name='pursuit_mlp_baseline')

    # init sampler
    if args.sampler == 'simple':
        sampler_cls = SimpleSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=4000, n_timesteps_max=64000, timestep_rate=40,
                            adaptive=False)
    elif args.sampler == 'parallel':
        sampler_cls = ParallelSampler
        sampler_args = dict(max_traj_len=args.max_traj_len, n_timesteps=args.n_timesteps,
                            n_timesteps_min=4000, n_timesteps_max=64000, timestep_rate=40,
                            adaptive=False, n_workers=args.sampler_workers)

    # init TRPO step function
    step_func = rltools.algos.policyopt.TRPO(max_kl=args.max_kl)

    # init optimizer
    popt = rltools.algos.policyopt.SamplingPolicyOptimizer(env=env, policy=policy,
                                                           baseline=baseline, step_func=step_func,
                                                           discount=args.discount,
                                                           gae_lambda=args.gae_lambda,
                                                           sampler_cls=sampler_cls,
                                                           sampler_args=sampler_args,
                                                           n_iter=args.n_iter)

    # initialize logger
    argstr = json.dumps(vars(args), separators=(',', ':'), indent=2)
    rltools.util.header(argstr)
    log_f = rltools.log.TrainingLog(args.log, [('args', argstr)])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        popt.train(sess, log_f, args.save_freq)


if __name__ == '__main__':
    main()
