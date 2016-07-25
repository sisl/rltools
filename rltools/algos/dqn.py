from __future__ import absolute_import, print_function

import random
import numpy as np

from rltools import util
from rltools.algos import RLAlgorithm
from rltools.samplers import evaluate


class DQN(RLAlgorithm):

    def __init__(self, env, q_func, target_q_func, target_update_step, eps,
                 obsfeat_fn=lambda obs: obs, max_experience_size=10000, traj_sim_len=500,
                 batch_size=64, discount=0.99, n_iter=100000, start_iter=0, store_paths=False,
                 whole_paths=True, double_dqn=False, duel_net=False, **kwargs):
        self.env = env
        self.q_func = q_func
        self.target_q_func = target_q_func
        self.target_update_step = target_update_step
        self.eps = eps
        self.obsfeat_fn = obsfeat_fn
        self.max_experience_size = max_experience_size
        self.traj_sim_len = traj_sim_len
        self.batch_size = batch_size
        self.discount = discount
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.store_paths = store_paths
        self.whole_paths = whole_paths

        self.double_dqn = double_dqn
        self.duel_net = duel_net

        self.memory = []
        self.total_time = 0.0

        self.debug = True

    def initialize(self, sess):
        self.target_q_func.copy_params_from_primary(sess)
        if self.debug:
            assert all(np.allclose(p1, p2)
                       for p1, p2 in util.safezip(
                           self.q_func.get_params(sess), self.target_q_func.get_params(sess)))

    def _compute_action(self, sess, obsfeat_Df):
        assert len(obsfeat_Df.shape) == 1
        if random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            return self.q_func.compute_qactions(sess, obsfeat_Df)[0, 0]

    def _pack_into_batch(self, transitions):
        num = len(transitions)
        obs_B_Do = np.zeros((num, self.env.observation_space.shape[0]))
        actions_B_Da = np.zeros((num, 1))
        rewards_B = np.zeros(num)
        succ_obs_B_Do = np.zeros((num, self.env.observation_space.shape[0]))
        for i, (obs_Do, action_Da, reward, succ_obs_Do) in enumerate(transitions):
            obs_B_Do[i, :] = obs_Do
            actions_B_Da[i, :] = action_Da
            rewards_B[i] = reward
            succ_obs_B_Do[i, :] = succ_obs_Do

        return obs_B_Do, actions_B_Da, rewards_B, succ_obs_B_Do

    def train(self, sess, log, save_freq):
        self.initialize(sess)
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0:
                log.write_snapshot(sess, self.q_func, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            curr_obs_Do = self.env.reset()
            q_loss = np.zeros(self.traj_sim_len)
            with util.Timer() as t_sim:
                for t in range(self.traj_sim_len):
                    curr_action_Da = self._compute_action(sess, self.obsfeat_fn(curr_obs_Do))
                    next_obs_Do, curr_reward, done, _ = self.env.step(curr_action_Da)

                    # TODO: not sure, maybe after adding to memory?
                    if done:
                        break

                    # Memory (s,a,r,s')
                    self.memory.append((curr_obs_Do, curr_action_Da, curr_reward, next_obs_Do))
                    self.memory = self.memory[-self.max_experience_size:]

                    curr_obs_Do = next_obs_Do

                    transitions_B = [self.memory[idx]
                                     for idx in np.random.choice(
                                         len(self.memory), size=self.batch_size)]
                    batch_obs_B_Do, batch_actions_B_Da, batch_rewards_B, batch_succ_obs_B_Do = self._pack_into_batch(
                        transitions_B)

                    batch_succ_target_actions_B_Da = self.target_q_func.compute_qactions(
                        sess, self.obsfeat_fn(batch_succ_obs_B_Do))
                    batch_qtargets_B = batch_rewards_B + self.discount * self.target_q_func.compute_qvals(
                        sess, self.obsfeat_fn(batch_succ_obs_B_Do), batch_succ_target_actions_B_Da)

                    q_loss[t] = self.q_func.opt_step(sess, self.obsfeat_fn(batch_obs_B_Do),
                                                     batch_actions_B_Da, batch_qtargets_B)

            if itr % self.target_update_step == self.target_update_step - 1:
                self.target_q_func.copy_params_from_primary(sess)

            with util.Timer() as t_eval:
                eval_return = evaluate(
                    self.env, self.obsfeat_fn,
                    lambda ofeat: self.q_func.compute_qactions(sess, ofeat)[0, 0],
                    self.traj_sim_len, 100)

        self.total_time += t_all.dt

        fields = [('iter', itr, int),
                  ('q_loss', q_loss.mean(), float),  # Average q loss
                  ('ret', eval_return, float),
                  ('tsim', t_sim.dt, float),  # Time for each traj simulation
                  ('teval', t_eval.dt, float),
                  ('ttotal', self.total_time, float)]
        return fields
