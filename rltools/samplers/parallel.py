import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rltools.samplers import Sampler, rollout
from rltools.trajutil import TrajBatch, Trajectory


class ThreadedSampler(Sampler):

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive=False, n_workers=4):
        super(ThreadedSampler, self).__init__(algo, max_traj_len, batch_size, min_batch_size,
                                              max_batch_size, batch_rate, adaptive)

        self.n_workers = n_workers

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.batch_size < self.max_batch_size:
            if itr % self.batch_rate == 0:
                self.batch_size *= 2

        r_func = lambda mtl: rollout(self.algo.env, self.algo.obsfeat_fn, lambda ofeat: self.algo.policy.sample_actions(sess, ofeat), mtl, self.algo.policy.action_space)

        with ThreadPoolExecutor(self.n_workers) as self.executor:
            trajs = self.executor.map(r_func, [self.max_traj_len] * self.batch_size)

        if not isinstance(trajs, list):
            trajs = list(trajs)

        trajbatch = TrajBatch.FromTrajs(trajs)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(),
                  float),  # average return for batch of traj
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])),
                  int),  # average traj length
                 ('ravg', trajbatch.r.stacked.mean(),
                  int)  # avg reward encountered per time step (probably not that useful)
                ])
