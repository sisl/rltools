from __future__ import absolute_import, print_function

import gym
import tensorflow as tf

import rltools.log
from rltools.algos.dqn import DQN
from rltools.qnet.categorical_qnet import CategoricalQFunction

SIMPLE_ARCH = '''[
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"},
        {"type": "fc", "n": 32},
        {"type": "nonlin", "func": "relu"}
    ]
    '''


def run():
    env = gym.make('CartPole-v0')
    q_func = CategoricalQFunction(env.observation_space, env.action_space, SIMPLE_ARCH, 1e-2,
                                  varscope_name='QFunc')
    target_q_func = CategoricalQFunction(env.observation_space, env.action_space, SIMPLE_ARCH, 1e-2,
                                         primary_q_func=q_func, varscope_name='targetQFunc')
    algorithm = DQN(env, q_func, target_q_func, 20, 0.01)
    log_f = rltools.log.TrainingLog(None, [])

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        algorithm.train(sess, log_f, 20)


if __name__ == '__main__':
    run()
