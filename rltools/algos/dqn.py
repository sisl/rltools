from __future__ import absolute_import, print_function

import tensorflow as tf

from rltools import optim, util
from rltools.algos import RLAlgorithm
from rltools.policy.stochastic import StochasticPolicy
from rltools.samplers.serial import ExperienceReplay
from rltools.samplers import evaluate

from copy import deepcopy

class DQNOptimizer(RLAlgorithm):

    def __init__(self, env, policy, 
                 obsfeat_fn=lambda obs: obs,
                 discount=0.99,
                 n_iter=500,
                 start_iter=0,
                 sample_cls=None, 
                 start_eps=1.0,
                 end_eps=0.05,
                 eval_freq=20,
                 n_eval_traj=50,
                 varscope_name="dqn",
                 sampler_cls=None,
                 sampler_args=dict(max_traj_len=200, batch_size=32, adaptive=False,
                                   min_batch_size=4, max_batch_size=64, batch_rate=40, initial_exploration=5000,
                                   max_experience=10000), **kwargs):

        self.env = env
        self.eval_env = deepcopy(env)
        self.policy = policy
        self.target_network = policy
        self.obsfeat_fn = obsfeat_fn
        self.discount = discount
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.eps = start_eps
        self.eps_step = (start_eps - end_eps) / n_iter
        self.eval_freq = eval_freq
        self.n_eval_traj = n_eval_traj
        if sampler_cls is None:
            sampler_cls = ExpereinceReplay
        self.sampler = sampler_cls(self, **sampler_args)

        # define training ops
        with tf.variable_scope
        #self.next_action_vals = self._next_act_score_op()

        self.total_time = 0.0

    def train(self, sess, log, save_freq):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            self.eps -= self.eps_step
            self.policy.set_eps(self.eps)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0:
                log.write_snapshot(sess, self.policy, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Take the policy grad step
            #with util.Timer() as t_step:
            #    params0_P = self.policy.get_params(sess)
            #    step_print_fields = self.step_func(sess, self.policy, trajbatch,
            #                                       trajbatch_vals['advantage'])
            #    self.policy.update_obsnorm(sess, trajbatch.obsfeat.stacked)

            if itr % self.eval_freq == 0:
                rave = evaluate(self.eval_env, self.obsfeat_fn, lambda ofeat: self.policy.deterministic_action(sess,
                    ofeat), self.sampler.max_traj_len, self.n_eval_traj)
                eval_fileds = [('r_eval', rave, float)]
            else:
                eval_fileds = [('r_eval', None, float)]


        # LOG
        self.total_time += t_all.dt

        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + eval_fileds + [
            ('eps', self.policy.eps, float)
        ]
        #  + step_print_fields + [
        #    ('tsamp', t_sample.dt, float),  # Time for sampling
        #    ('tbase', t_base.dt, float),  # Time for advantage/baseline computation
        #    ('tstep', t_step.dt, float),
        #    ('ttotal', self.total_time, float)
        #]
        return fields

    def _next_act_score_op(self):
        return tf.stop_gradient(self.target_network(self.next_observation))

    def dqn_update(self, sess, samples):
        feed = {self.observations: samples['observations'],
                self.next_observations: samples['next_observations']}
