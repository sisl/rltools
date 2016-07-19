import copy
import random

from gym import spaces
import numpy as np

from rltools.samplers import Sampler, rollout
from rltools.trajutil import TrajBatch, Trajectory


class SimpleSampler(Sampler):

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive=False):
        super(SimpleSampler, self).__init__(algo, max_traj_len, batch_size, min_batch_size,
                                            max_batch_size, batch_rate, adaptive)

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.batch_size < self.max_batch_size:
            if itr % self.batch_rate == 0:
                self.batch_size *= 2

        trajs = []
        for _ in range(self.batch_size):
            obs, obsfeat, actions, actiondists, rewards = rollout(self.algo.env,
                                                                 self.algo.obsfeat_fn,
                                                                 lambda ofeat: self.algo.policy.sample_actions(sess, ofeat),
                                                                 self.max_traj_len, self.algo.policy.action_space)

            obs_T_Do = np.concatenate(obs)
            assert obs_T_Do.shape[0] == len(obs), '{} != {}'.format(obs_T_Do.shape, len(obs))
            obsfeat_T_Df = np.concatenate(obsfeat)
            assert obsfeat_T_Df.shape[0] == len(obs), '{} != {}'.format(obsfeat_T_Df.shape,
                                                                        len(obs))
            adist_T_Pa = np.concatenate(actiondists)
            assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
            a_T_Da = np.concatenate(actions)
            assert a_T_Da.shape[0] == len(obs)
            r_T = np.asarray(rewards)
            assert r_T.shape == (len(obs),)
            trajs.append(Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T))
        trajbatch = TrajBatch.FromTrajs(trajs)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(),
                  float),  # average return for batch of traj
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])),
                  int),  # average traj length
                 ('ravg', trajbatch.r.stacked.mean(), int
                 )  # avg reward encountered per time step (probably not that useful)
                ])


class DecSampler(Sampler):

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive=False):
        super(DecSampler, self).__init__(algo, max_traj_len, batch_size, min_batch_size,
                                         max_batch_size, batch_rate, adaptive)

    def sample(self, sess, itr):
        assert self.batch_size >= self.algo.env.n_agents(
        ), 'Batch size should be at least as large as number of agents'
        assert self.batch_size % self.algo.env.n_agents(
        ) == 0, 'Batch size should be evenly divisible by number of agents'
        if self.adaptive and itr > 0 and self.batch_size < self.max_batch_size:
            if itr % self.batch_rate == 0:
                self.batch_size *= 2

        def get_lists(nl, na):
            l = []
            for i in range(nl):
                l.append([[] for j in xrange(na)])
            return l

        env = self.algo.env
        trajs = []
        for _ in range(self.batch_size /
                       env.n_agents()):  #FIXME: batch size depends on number of agents
            old_ob = env.reset()
            n_total = env.n_agents()
            obs, obsfeat, actions, actiondists, rewards = get_lists(5, n_total)
            for itr in range(self.max_traj_len):
                agent_actions = []
                n = env.n_agents()
                for i, agent_obs in enumerate(old_ob):
                    if agent_obs is None:
                        continue
                    obs[i].append(np.expand_dims(agent_obs, 0))
                    obsfeat[i].append(self.algo.obsfeat_fn(obs[i][-1]))
                    a, adist = self.algo.policy.sample_actions(sess, obsfeat[i][-1])
                    agent_actions.append(a)
                    actions[i].append(a)
                    actiondists[i].append(adist)
                    new_ob, r, done, _ = env.step(np.array(agent_actions)[:, 0, 0])  #FIXME
                for i, o in enumerate(old_ob):
                    if o is None:
                        continue
                    rewards[i].append(r)
                    old_ob = new_ob
                if done:
                    break

            for agnt in range(n_total):
                obs_T_Do = np.concatenate(obs[agnt])
                obsfeat_T_Df = np.concatenate(obsfeat[agnt])
                adist_T_Pa = np.concatenate(actiondists[agnt])
                a_T_Da = np.concatenate(actions[agnt])
                r_T = np.asarray(rewards[agnt])
                trajs.append(Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T))

        trajbatch = TrajBatch.FromTrajs(trajs)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(),
                  float),  # average return for batch of traj
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])),
                  int),  # average traj length
                 ('ravg', trajbatch.r.stacked.mean(), int
                 )  # avg reward encountered per time step (probably not that useful)
                ])


class ImportanceWeightedSampler(SimpleSampler):
    """
    Alternate between sampling iterations using simple sampler and importance sampling iterations

    Does not work with a NN value function baseline
    """

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive=False, n_backtrack='all', randomize_draw=False, n_pretrain=0,
                 skip_is=False, max_is_ratio=0):
        """
        n_backtrack: number of past policies to update from
        n_pretrain: iteration number until which to only do importance sampling
        skip_is: whether to skip doing alternate importance sampling after pretraining
        max_is_ratio: maximum importance sampling ratio (thresholding)
        """
        self.n_backtrack = n_backtrack
        self.randomize_draw = randomize_draw
        self.n_pretrain = n_pretrain
        self.skip_is = skip_is
        self.max_is_ratio = max_is_ratio
        self._hist = []
        self._is_itr = 0
        super(ImportanceWeightedSampler, self).__init__(
            algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive)
        assert not self.adaptive, "Can't use adaptive sampling with importance weighted for now"  # TODO needed?

    @property
    def history(self):
        return self._hist

    def add_history(self, trajbatch):
        self.history.append(trajbatch)

    def get_history(self, n_past='all'):
        if n_past == 'all':
            return self.history
        assert isinstance(n_past, int)
        return self.history[-min(n_past, len(self.history)):]

    def sample(self, sess, itr):
        # Importance sampling for first few iterations
        if itr < self.n_pretrain:
            trajbatch = self.is_sample(sess, itr)
            return trajbatch

        # Alternate between importance sampling and actual sampling
        # Data logs will be messy TODO
        if self._is_itr and not self.skip_is:
            trajbatch, batch_info = self.is_sample(sess, itr)
        else:
            trajbatch, batch_info = super(ImportanceWeightedSampler, self).sample(sess, itr)
            if not self.skip_is:
                self.add_history(trajbatch)

        self._is_itr = (self._is_itr + 1) % 2

        return trajbatch, batch_info

    def is_sample(self, sess, itr):
        rettrajs = []
        for hist_trajbatch in self.get_history(self.n_backtrack):
            n_trajs = len(hist_trajbatch)
            n_samples = min(n_trajs, self.batch_size)

            if self.randomize_draw:
                samples = random.sample(hist_trajbatch, n_samples)
            elif hist_trajbatch:
                # Random start
                start = random.randint(0, n_trajs - n_samples)
                samples = hist_trajbatch[start:start + n_samples]

            samples = copy.deepcopy(samples)  # Avoid overwriting

            for traj in samples:
                # What the current policy would have done
                _, adist_T_Pa = self.algo.policy.sample_actions(sess, traj.obsfeat_T_Df)
                # What the older policy did
                hist_adist_T_Pa = traj.adist_T_Pa

                assert traj.adist_T_Pa.shape == adist_T_Pa.shape
                # Use newer policy distribution
                traj.adist_T_Pa = adist_T_Pa

                # Log probabilities of actions using previous and current
                logprob_curr = self.algo.policy.distribution.log_density(adist_T_Pa, traj.a_T_Da)
                logprob_hist = self.algo.policy.distribution.log_density(hist_adist_T_Pa,
                                                                         traj.a_T_Da)
                # Importance sampling ratio
                is_ratio = np.exp(logprob_curr.sum() - logprob_hist.sum())

                # Thresholding
                if self.max_is_ratio > 0:
                    is_ratio = min(is_ratio, self.max_is_ratio)

                # Weight the rewards accordingly
                traj.r_T *= is_ratio

            rettrajs.extend(samples)
            # Pack them back
        if len(rettrajs) > self.batch_size:
            rettrajs = random.sample(rettrajs, self.batch_size)
            rettrajbatch = TrajBatch.FromTrajs(rettrajs)

        batch_info = [('ret', rettrajbatch.r.padded(fill=0.).sum(axis=1).mean(),
                       float),  # average return for batch of traj
                      ('avglen', int(np.mean([len(traj) for traj in rettrajbatch])),
                       int),  # average traj length
                      ('ravg', rettrajbatch.r.stacked.mean(), int
                      )  # avg reward encountered per time step (probably not that useful)
                     ]

        return rettrajbatch, batch_info
