import tensorflow as tf

from rltools.algos import RLAlgorithm
from rltools.policy.deterministic import DeterministicPolicy
from rltools.samplers.serial import SimpleSampler


class DDPG(RLAlgorithm):

    def __init__(self, env, policy, target_policy, q_func, target_q_func, target_update_rate,
                 exploration_noise_std, obsfeat_fn=lambda obs: obs, discount=0.99, n_iter=500,
                 start_iter=0, store_paths=False, whole_paths=True, sampler_cls=None,
                 sampler_args=dict(max_traj_len=200, batch_size=32, adaptive=False,
                                   min_batch_size=4, max_batch_size=64, batch_rate=40), **kwargs):
        self.env = env
        self.policy = policy
        self.target_policy = target_policy
        self.q_func = q_func
        self.target_q_func = target_q_func
        self.target_update_rate = target_update_rate
        self.exploration_noise_std = exploration_noise_std
        self.obsfeat_fn = obsfeat_fn
        self.discount = discount
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        if sampler_cls is None:
            sampler_cls = SimpleSampler
        self.sampler = sampler_cls(self, **sampler_args)

        # Experience memory
        self.memory = []

        self.total_time = 0.0

    def train(self, sess, log, save_freq):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0:
                log.write_snapshot(sess, self.policy, itr)
                log.write_snapshot(sess, self.q_func, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            curr_obs_Do = self.env.reset()
            q_loss = np.zeros(self.traj_sim_len)
            with util.Timer() as t_sim:
                for t in range(self.traj_sim_len):
                    curr_action_Da = self.policy.compute_actions(sess, self.obsfeat_fn(curr_obs_Do))
                    curr_action_Da += np.random.rand(
                        *curr_action_Da.shape) * self.exploration_noise_std

                    next_obs_Do, reward, done, _ = self.env.step(curr_action_Da)

                    # Memory (s,a,r,s')
                    self.memory.append(
                        (curr_obs_Do, curr_action_Da, curr_reward, next_obs_Do, int(done)))
                    self.memory = self.memory[-self.max_experience_size:]

                    curr_obs_Do = next_obs_Do

                    transitions_B = [self.memory[idx]
                                     for idx in np.random.choice(
                                         len(self.memory), size=self.batch_size)]
                    batch_obs_B_Do, batch_actions_B_Da, batch_rewards_B, batch_succ_obs_B_Do, batch_done_B = self._pack_into_batch(
                        transitions_B)

                    batch_targetpolicy_succ_actions_B_Da = self.target_policy.compute_actions(
                        sess, self.obsfeat_fn(batch_succ_obs_B_Do))
                    batch_qtargets_B = batch_rewards_B + self.discount * self.target_q_func.compute_qvals(
                        sess, self.obsfeat_fn(batch_succ_obs_B_Do),
                        batch_targetpolicy_succ_actions_B_Da)
                    assert batch_qtargets_B.shape == (self.batch_size,)

                    q_loss[t] = self.q_func.opt_step(sess, self.obsfeat_fn(batch_obs_B_Do),
                                                     batch_actions_B_Da, batch_qtargets_B)

                    batch_currpolicy_actions_B_Da = self.policy.compute_actions(
                        sess, self.obsfeat_fn(batch_obs_B_Do))
                    batch_qgrads_B_Da = self.q_func.eval_grad_wrt_action(
                        sess, self.obsfeat_fn(batch_obs_B_Do), batch_currpolicy_actions_B_Da)
                    assert batch_qgrads_B_Da.shape == (self.batch_size,
                                                       self.policy.action_space.shape[0])

                    # Policy gradients
                    # TODO

                    # Policy gradient step
                    self.policy.descent_step(sess, batch_policygrad_Dp)

                    self.target_policy.interp_params_with_primary(sess, self.target_update_rate)
                    self.target_q_func.interp_params_with_primary(sess, self.target_update_rate)
