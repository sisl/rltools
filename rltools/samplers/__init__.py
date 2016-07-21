import numpy as np
from gym import spaces

import rltools.util
from rltools.trajutil import RaggedArray, TrajBatch, Trajectory


class Sampler(object):
    """
    Base Sampler class
    """

    def __init__(self, algo, max_traj_len, batch_size, min_batch_size, max_batch_size, batch_rate,
                 adaptive):
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
        assert not self.algo.discount is None
        qvals_zfilled_B_T = rltools.util.discount(rewards_B_T, self.algo.discount)
        assert qvals_zfilled_B_T.shape == (self.batch_size, maxT)
        q = RaggedArray([qvals_zfilled_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])
        q_B_T = q.padded(fill=np.nan)  # q vals padded with nans in the end
        assert q_B_T.shape == (self.batch_size, maxT)

        # Time-dependent baseline
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (self.batch_size, 1))
        assert simplev_B_T.shape == (self.batch_size, maxT)
        simplev = RaggedArray([simplev_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])

        # State-dependent baseline
        v_stacked = self.algo.baseline.predict(sess, trajbatch)
        assert v_stacked.ndim == 1
        v = RaggedArray(v_stacked, lengths=trajlens)

        # Compare squared loss of value function to that of time-dependent value function
        # Explained variance
        # *_r2 = 1 - var(y-ypred)/var(y)
        # *_r2 = 0 => Useless
        # *_r2 = 1 => Perfect
        # *_r2 < 0 => Worse than useless
        constfunc_prediction_loss = np.var(q.stacked)
        simplev_prediction_loss = np.var(q.stacked - simplev.stacked)
        simplev_r2 = 1. - simplev_prediction_loss / (constfunc_prediction_loss + 1e-8)
        vfunc_prediction_loss = np.var(q.stacked - v_stacked)
        vfunc_r2 = 1. - vfunc_prediction_loss / (constfunc_prediction_loss + 1e-8)

        # XXX HACK
        if vfunc_r2 < 0:
            v = simplev

        # Compute advantage -- GAE(gamma,lambda) estimator
        v_B_T = v.padded(fill=0.)
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((self.batch_size, 1))], axis=1)
        assert v_B_Tp1.shape == (self.batch_size, maxT + 1)
        delta_B_T = rewards_B_T + self.algo.discount * v_B_Tp1[:, 1:] - v_B_Tp1[:, :-1]
        adv_B_T = rltools.util.discount(delta_B_T, self.algo.discount * self.algo.gae_lambda)
        assert adv_B_T.shape == (self.batch_size, maxT)
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(trajlens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)

        # Fit for the next time step
        baseline_info = self.algo.baseline.fit(sess, trajbatch, q.stacked)

        return dict(advantage=adv, qval=q, v_r=vfunc_r2, tv_r=simplev_r2), baseline_info

    def stop(self):
        raise NotImplementedError()


def rollout(env, obsfeat_fn, act_fn, max_traj_len, action_space):
    obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
    obs.append((env.reset())[None, ...].copy())

    for itr in range(max_traj_len):
        obsfeat.append(obsfeat_fn(obs[-1]))
        a, adist = act_fn(obsfeat[-1])
        actions.append(a)
        actiondists.append(adist)
        if isinstance(action_space, spaces.Discrete):
            assert a.ndim == 2 and a.dtype in (np.int32, np.int64)
            if hasattr(action_space, 'ndim'):
                o2, r, done, _ = env.step(actions[-1][0, :action_space.ndim])
            else:
                o2, r, done, _ = env.step(actions[-1][0, 0])
        else:
            o2, r, done, _ = env.step(actions[-1])

        rewards.append(r)
        if done:
            break
        if itr != max_traj_len - 1:
            obs.append(o2[None, ...])

    obs_T_Do = np.concatenate(obs)
    assert obs_T_Do.shape[0] == len(obs), '{} != {}'.format(obs_T_Do.shape, len(obs))
    obsfeat_T_Df = np.concatenate(obsfeat)
    assert obsfeat_T_Df.shape[0] == len(obs), '{} != {}'.format(obsfeat_T_Df.shape, len(obs))
    adist_T_Pa = np.concatenate(actiondists)
    assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
    a_T_Da = np.concatenate(actions)
    # TODO: for facotred policy assertion fails 
    #assert a_T_Da.shape[0] == len(obs)
    r_T = np.asarray(rewards)
    assert r_T.shape == (len(obs),)

    return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)
