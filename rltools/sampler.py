import copy
import random

import numpy as np

import util


class Trajectory(object):
    __slots__ = ('obs_T_Do', 'obsfeat_T_Df', 'adist_T_Pa', 'a_T_Da', 'r_T')
    def __init__(self, obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T):
        assert (
            obs_T_Do.ndim > 1 and obsfeat_T_Df.ndim > 1 and adist_T_Pa.ndim > 1 and a_T_Da.ndim > 1 and
            r_T.ndim == 1 and
            obs_T_Do.shape[0] == obsfeat_T_Df.shape[0] == adist_T_Pa.shape[0] == a_T_Da.shape[0] == r_T.shape[0]
        )
        self.obs_T_Do = obs_T_Do
        self.obsfeat_T_Df = obsfeat_T_Df
        self.adist_T_Pa = adist_T_Pa
        self.a_T_Da = a_T_Da
        self.r_T = r_T

    def __len__(self):
        return self.obs_T_Do.shape[0]

    # Saving/loading discards obsfeat
    def save_h5(self, grp, **kwargs):
        grp.create_dataset('obs_T_Do', data=self.obs_T_Do, **kwargs)
        grp.create_dataset('adist_T_Pa', data=self.adist_T_Pa, **kwargs)
        grp.create_dataset('a_T_Da', data=self.a_T_Da, **kwargs)
        grp.create_dataset('r_T', data=self.r_T, **kwargs)

    @classmethod
    def LoadH5(cls, grp, obsfeat_fn):
        """
        obsfeat_fn: Used to fill in observation features.
                    If None, the raw observations will be copied over.
        """
        obs_T_Do = grp['obs_T_Do'][...]
        obsfeat_T_Df = obsfeat_fn(obs_T_Do) if obsfeat_fn is not None else obs_T_Do.copy()
        return cls(obs_T_Do, obsfeat_T_Df, grp['adist_T_Pa'][...], grp['a_T_Da'][...], grp['r_T'][...])


def raggedstack(arrays, fill=0., axis=0, raggedaxis=1):
    """
    Stacks a list of arrays, like np.stack with axis=0.
    Arrays may have different length (along the raggedaxis), and will be padded on the right
    with the given fill value.
    """
    assert axis == 0 and raggedaxis == 1, 'not implemented'
    arrays = [a[None,...] for a in arrays]
    assert all(a.ndim >= 2 for a in arrays)

    outshape = list(arrays[0].shape)
    outshape[0] = sum(a.shape[0] for a in arrays)
    outshape[1] = max(a.shape[1] for a in arrays) # take max along ragged axes
    outshape = tuple(outshape)

    out = np.full(outshape, fill, dtype=arrays[0].dtype)
    pos = 0
    for a in arrays:
        out[pos:pos+a.shape[0], :a.shape[1], ...] = a
        pos += a.shape[0]
    assert pos == out.shape[0]
    return out


class RaggedArray(object):
    def __init__(self, arrays, lengths=None):
        if lengths is None:
            # Without provided lengths, `arrays` is interpreted as a list of arrays
            # and self.lengths is set to the list of lengths for those arrays
            self.arrays = arrays
            self.stacked = np.concatenate(arrays, axis=0)
            self.lengths = np.array([len(a) for a in arrays])
        else:
            # With provided lengths, `arrays` is interpreted as concatenated data
            # and self.lengths is set to the provided lengths.
            self.arrays = np.split(arrays, np.cumsum(lengths)[:-1])
            self.stacked = arrays
            self.lengths = np.asarray(lengths, dtype=int)
            assert all(len(a) == l for a,l in util.safezip(self.arrays, self.lengths))
            self.boundaries = np.concatenate([[0], np.cumsum(self.lengths)])
            assert self.boundaries[-1] == len(self.stacked)
    def __len__(self):
        return len(self.lengths)
    def __getitem__(self, idx):
        return self.stacked[self.boundaries[idx]:self.boundaries[idx+1], ...]
    def padded(self, fill=0.):
        return raggedstack(self.arrays, fill=fill, axis=0, raggedaxis=1)


class TrajBatch(object):
    def __init__(self, trajs, obs, obsfeat, adist, a, r, time):
        self.trajs, self.obs, self.obsfeat, self.adist, self.a, self.r, self.time = trajs, obs, obsfeat, adist, a, r, time

    @classmethod
    def FromTrajs(cls, trajs):
        assert all(isinstance(traj, Trajectory) for traj in trajs)
        obs = RaggedArray([t.obs_T_Do for t in trajs])
        obsfeat = RaggedArray([t.obsfeat_T_Df for t in trajs])
        adist = RaggedArray([t.adist_T_Pa for t in trajs])
        a = RaggedArray([t.a_T_Da for t in trajs])
        r = RaggedArray([t.r_T for t in trajs])
        time = RaggedArray([np.arange(len(t), dtype=float) for t in trajs])
        return cls(trajs, obs, obsfeat, adist, a, r, time)

    def with_replaced_reward(self, new_r):
        new_trajs = [Trajectory(traj.obs_T_Do, traj.obsfeat_T_Df, traj.adist_T_Pa, traj.a_T_Da, traj_new_r) for traj, traj_new_r in util.safezip(self.trajs, new_r)]
        return TrajBatch(new_trajs, self.obs, self.obsfeat, self.adist, self.a, new_r, self.time)

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        return self.trajs[idx]

    def save_h5(self, f, starting_id=0, **kwargs):
        for i, traj in enumerate(self.trajs):
            traj.save_h5(f.require_group('%06d' % (i+starting_id)), **kwargs)

    @classmethod
    def LoadH5(cls, dset, obsfeat_fn):
        return cls.FromTrajs([Trajectory.LoadH5(v, obsfeat_fn) for k, v in dset.iteritems()])


class Sampler(object):
    """
    Base Sampler class
    """
    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive):
        self.algo = algo
        self.max_traj_len = max_traj_len
        self.adaptive = adaptive
        self.batch_size = batch_size
        if self.adaptive:
            self.batch_size = min_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_rate = batch_rate


    def start(self):
        """Init sampler"""
        raise NotImplementedError()

    def sample(self, itr):
        """Collect samples"""
        raise NotImplementedError()

    def process(self, sess, itr, trajbatch):
        assert len(trajbatch) == self.batch_size
        trajlens = [len(traj) for traj in trajbatch]
        maxT = max(trajlens)

        rewards_B_T = trajbatch.r.padded(fill=0.)
        qvals_zfilled_B_T = util.discount(rewards_B_T, self.algo.discount); assert qvals_zfilled_B_T.shape == (self.batch_size, maxT)
        q = RaggedArray([qvals_zfilled_B_T[i,:len(traj)] for i, traj in enumerate(trajbatch)])
        q_B_T = q.padded(fill=np.nan) # q vals padded with nans in the end
        assert q_B_T.shape == (self.batch_size, maxT)

        # Time-dependent baseline
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (self.batch_size, 1)); assert simplev_B_T.shape == (self.batch_size, maxT)
        simplev = RaggedArray([simplev_B_T[i,:len(traj)] for i, traj in enumerate(trajbatch)])

        # State-dependent baseline
        v_stacked = self.algo.baseline.predict(sess, trajbatch); assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=trajlens)

        # Compare squared loss of value function to that of time-dependent value function
        # Explained variance
        # *_r2 = 1 - var(y-ypred)/var(y)
        # *_r2 = 0 => Useless
        # *_r2 = 1 => Perfect
        # *_r2 <0 => Worse than useless
        constfunc_prediction_loss = np.var(q.stacked)
        simplev_prediction_loss = np.var(q.stacked-simplev.stacked)
        simplev_r2 = 1. - simplev_prediction_loss/(constfunc_prediction_loss + 1e-8)
        vfunc_prediction_loss = np.var(q.stacked-v_stacked)
        vfunc_r2 = 1. - vfunc_prediction_loss/(constfunc_prediction_loss + 1e-8)

        # Compute advantage -- GAE(gamma,lambda) estimator
        v_B_T = v.padded(fill=0.)
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((self.batch_size,1))], axis=1); assert v_B_Tp1.shape == (self.batch_size, maxT+1)
        delta_B_T = rewards_B_T + self.algo.discount*v_B_Tp1[:,1:] - v_B_Tp1[:,:-1]
        adv_B_T = util.discount(delta_B_T, self.algo.discount*self.algo.gae_lambda); assert adv_B_T.shape == (self.batch_size, maxT)
        adv = RaggedArray([adv_B_T[i,:l] for i,l in enumerate(trajlens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)

        # Fit for the next time step
        baseline_info = self.algo.baseline.fit(sess, trajbatch, q.stacked)

        return dict(advantage=adv, qval=q, v_r=vfunc_r2, tv_r=simplev_r2), baseline_info


    def stop(self):
        raise NotImplementedError()


class SimpleSampler(Sampler):
    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive=False):
        super(SimpleSampler, self).__init__(algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive)

    def sample(self, sess, itr):
        if self.adaptive and itr > 0 and self.batch_size < self.max_batch_size:
            if itr % self.batch_rate == 0:
                self.batch_size *= 2

        trajs = []
        for _ in range(self.batch_size):
            obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
            obs.append((self.algo.env.reset())[None,...].copy())
            for itr in range(self.max_traj_len):
                obsfeat.append(self.algo.obsfeat_fn(obs[-1]))
                a, adist = self.algo.policy.sample_actions(sess, obsfeat[-1])
                actions.append(a)
                actiondists.append(adist)
                o2, r, done, _ = self.algo.env.step(actions[-1][0,0]) # FIXME
                rewards.append(r)
                if done:
                    break
                if itr!=self.max_traj_len-1:
                    obs.append(o2[None,...])

            obs_T_Do = np.concatenate(obs); assert obs_T_Do.shape[0] == len(obs), '{} != {}'.format(obs_T_Do.shape, len(obs))
            obsfeat_T_Df = np.concatenate(obsfeat); assert obsfeat_T_Df.shape[0] == len(obs), '{} != {}'.format(obsfeat_T_Df.shape, len(obs))
            adist_T_Pa = np.concatenate(actiondists); assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
            a_T_Da = np.concatenate(actions); assert a_T_Da.shape == (len(obs), 1)
            r_T = np.asarray(rewards); assert r_T.shape == (len(obs),)
            trajs.append(Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T))
        trajbatch = TrajBatch.FromTrajs(trajs)
        return (trajbatch,
                [('ret', trajbatch.r.padded(fill=0.).sum(axis=1).mean(), float), # average return for batch of traj
                 ('avglen', int(np.mean([len(traj) for traj in trajbatch])), int), # average traj length
                 ('ravg', trajbatch.r.stacked.mean(), int) # avg reward encountered per time step (probably not that useful)
                ])


class BatchSampler(Sampler):
    def __init__(self, algo):
        self.algo = algo

    def sample(self, itr):
        # Completed trajs
        num_sa = 0
        completed_trajlists = []

        # Simulations and their current trajectories
        # TODO


class ImportanceWeightedSampler(SimpleSampler):
    """Alternate between sampling iterations using simple sampler and importance sampling iterations"""
    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive=False, n_backtrack='all'):
        """
        n_backtrack: number of past policies to update from
        """
        self.n_backtrack = n_backtrack
        self._hist = []
        self._is_itr = 0
        super(ImportanceWeightedSampler, self).__init__(algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate, adaptive)
        assert not self.adaptive, "Can't use adaptive sampling with importance weighted for now" # TODO needed?

    @property
    def history(self):
        return self._hist

    def add_history(self, trajbatch):
        self.history.append(trajbatch)

    def get_history(self, n_past='all'):
        if n_past == 'all':
            return self.history
        assert isinstance(n_past, int)
        return self.history[-min(n_past, len(self.history))]

    def sample(self, sess, itr):
        # Alternate between importance sampling and actual sampling
        # Data logs will be messy TODO
        if self._is_itr:
            trajbatch = self.is_sample(sess, itr)
        else:
            trajbatch = super(ImportanceWeightedSampler, self).sample(sess, itr)
            self.add_history(trajbatch)

        self._is_itr = (self._is_itr + 1) % 2

        return trajbatch

    def is_sample(self, sess, itr, randomize_draw=True):
        rettrajs = []
        for hist_trajbatch in self.get_history(self.n_backtrack):
            n_trajs = len(hist_trajbatch)
            n_samples = min(n_trajs, self.batch_size)

            if randomize_draw:
                samples = random.sample(hist_trajbatch, n_samples)
            elif hist_trajbatch:
                samples = hist_trajbatch[:n_samples]

            samples = copy.deepcopy(samples) # Avoid overwriting
            for traj in samples:
                # What the current policy would have done
                _, adist_T_Pa = self.algo.policy.sample_actions(sess, traj.obsfeat_T_Df)
                # What the older policy did
                hist_adist_T_Pa = traj.adist_T_Pa
                assert traj.adist_T_Pa.shape == adist_T_Pa
                # Use newer policy distribution
                traj.adist_T_Pa = adist_T_Pa

                # Log probabilities of actions using previous and current
                logprob_curr = self.algo.policy.distribution.log_density(adist_T_Pa, traj.a_T_Da)
                logprob_hist = self.algo.policy.distribution.log_density(hist_adist_T_Pa, traj.a_T_Da)
                # Importance sampling ratio
                is_ratio = np.exp(logprob_curr.sum() - logprob_hist.sum())
                # Weight the rewards accordingly
                traj.r_T *= is_ratio

            rettrajs.append(samples)
        # Pack them back
        rettrajbatch = TrajBatch.FromTrajs(rettrajs)
        if len(rettrajbatch) > self.batch_size:
            rettrajbatch = random.sample(rettrajbatch, self.batch_size)
        return rettrajbatch
