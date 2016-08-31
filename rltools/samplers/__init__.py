import numpy as np
from gym import spaces

from rltools import nn
import rltools.util
from rltools.trajutil import RaggedArray, TrajBatch, Trajectory


class Sampler(object):
    """
    Base Sampler class
    """

    def __init__(self, algo, n_timesteps, max_traj_len, timestep_rate, n_timesteps_min,
                 n_timesteps_max, adaptive):
        self.algo = algo
        self.n_timesteps = n_timesteps
        self.max_traj_len = max_traj_len
        self.adaptive = adaptive
        if self.adaptive:
            self.timestep_rate = timestep_rate
            self.n_timesteps_min = n_timesteps_min
            self.n_timesteps_max = n_timesteps_max
        self.n_episodes = 0

    def start(self):
        """Init sampler"""
        raise NotImplementedError()

    def sample(self, itr):
        """Collect samples"""
        raise NotImplementedError()

    def process(self, sess, itr, trajbatch, discount, gae_lambda, baseline):
        B = len(trajbatch)
        trajlens = [len(traj) for traj in trajbatch]
        maxT = max(trajlens)

        rewards_B_T = trajbatch.r.padded(fill=0.)
        assert not self.algo.discount is None
        qvals_zfilled_B_T = rltools.util.discount(rewards_B_T, discount)
        assert qvals_zfilled_B_T.shape == (B, maxT)
        q = RaggedArray([qvals_zfilled_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])
        q_B_T = q.padded(fill=np.nan)  # q vals padded with nans in the end
        assert q_B_T.shape == (B, maxT)

        # Time-dependent baseline
        simplev_B_T = np.tile(np.nanmean(q_B_T, axis=0, keepdims=True), (B, 1))
        assert simplev_B_T.shape == (B, maxT)
        simplev = RaggedArray([simplev_B_T[i, :len(traj)] for i, traj in enumerate(trajbatch)])

        # State-dependent baseline
        v_stacked = baseline.predict(trajbatch)
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
        v_B_Tp1 = np.concatenate([v_B_T, np.zeros((B, 1))], axis=1)
        assert v_B_Tp1.shape == (B, maxT + 1)
        delta_B_T = rewards_B_T + discount * v_B_Tp1[:, 1:] - v_B_Tp1[:, :-1]
        adv_B_T = rltools.util.discount(delta_B_T, discount * gae_lambda)
        assert adv_B_T.shape == (B, maxT)
        adv = RaggedArray([adv_B_T[i, :l] for i, l in enumerate(trajlens)])
        assert np.allclose(adv.padded(fill=0), adv_B_T)

        # Fit for the next time step
        baseline_info = baseline.fit(trajbatch, q.stacked)

        return dict(advantage=adv,
                    qval=q,
                    v_r=vfunc_r2,
                    tv_r=simplev_r2,), baseline_info

    def stop(self):
        raise NotImplementedError()


def centrollout(env, obsfeat_fn, act_fn, max_traj_len, action_space):
    assert env.reward_mech == 'global'
    obs, obsfeat, actions, actiondists, rewards = [], [], [], [], []
    obs.append(np.c_[env.reset()].ravel()[None, ...].copy())

    for itr in range(max_traj_len):
        obsfeat.append(obsfeat_fn(obs[-1]))
        a, adist = act_fn(obsfeat[-1])
        actions.append(a)
        actiondists.append(adist)
        if isinstance(action_space, spaces.Discrete):
            ndim = 1 if not hasattr(action_space, 'ndim') else action_space.ndim
            assert a.ndim == 2 and a.dtype in (np.int32, np.int64)
            if ndim == 1:
                o2, r, done, _ = env.step(actions[-1][0, 0])  # XXX
            else:
                o2, r, done, _ = env.step(actions[-1][0, :ndim])  # XXX
        else:
            o2, r, done, _ = env.step(actions[-1][0])

        assert (r == r[0]).all()
        rewards.append(r[0])
        if done:
            break
        if itr != max_traj_len - 1:
            obs.append(np.c_[o2].ravel()[None, ...])

    obs_T_Do = np.concatenate(obs)
    assert obs_T_Do.shape[0] == len(obs), '{} != {}'.format(obs_T_Do.shape, len(obs))
    obsfeat_T_Df = np.concatenate(obsfeat)
    assert obsfeat_T_Df.shape[0] == len(obs), '{} != {}'.format(obsfeat_T_Df.shape, len(obs))
    adist_T_Pa = np.concatenate(actiondists)
    assert adist_T_Pa.ndim == 2 and adist_T_Pa.shape[0] == len(obs)
    a_T_Da = np.concatenate(actions)
    assert a_T_Da.shape[0] == len(obs)
    r_T = np.asarray(rewards)
    assert r_T.shape == (len(obs),)

    return Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T)


def get_lists(nl, na):
    l = []
    for i in range(nl):
        l.append([[] for j in range(na)])
    return l


def decrollout(env, obsfeat_fn, act_fn, max_traj_len, action_space):
    # if not isinstance(act_fn, list):
    #     act_fn = [act_fn for _ in env.agents]
    # assert len(act_fn) == len(env.agents)
    # XXX
    assert not isinstance(act_fn, list)

    trajs = []
    old_obs = env.reset()
    obs, obsfeat, actions, actiondists, rewards = get_lists(5, len(env.agents))

    for itr in range(max_traj_len):
        agent_actions, adist_list = act_fn(np.asarray(old_obs))
        for i, agent_obs in enumerate(old_obs):
            obs[i].append(np.expand_dims(agent_obs, 0))
            obsfeat[i].append(obsfeat_fn(obs[i][-1]))
            actions[i].append(agent_actions[i])
            actiondists[i].append(adist_list[i])

        comp_actions = np.array(agent_actions)
        if isinstance(action_space, spaces.Discrete):
            new_obs, r, done, _ = env.step(comp_actions[:, 0, 0])
        else:
            new_obs, r, done, _ = env.step(comp_actions)

        for i, o in enumerate(old_obs):
            if o is None:
                continue
            rewards[i].append(r[i])
        old_obs = new_obs

        if done:
            break

    for agnt in range(len(env.agents)):
        obs_T_Do = np.concatenate(obs[agnt])
        obsfeat_T_Df = np.concatenate(obsfeat[agnt])
        adist_T_Pa = np.concatenate(np.expand_dims(np.asarray(actiondists[agnt]), 0))
        a_T_Da = np.concatenate(np.expand_dims(np.asarray(actions[agnt]), 0))
        r_T = np.asarray(rewards[agnt])
        trajs.append(Trajectory(obs_T_Do, obsfeat_T_Df, adist_T_Pa, a_T_Da, r_T))

    return trajs


def evaluate(env, obsfeat_fn, action_fn, max_traj_len, n_traj):
    rs = np.zeros(n_traj)
    for t in xrange(n_traj):
        rtot = 0.0
        o = env.reset()
        for itr in range(max_traj_len):
            a = action_fn(obsfeat_fn(np.expand_dims(o, 0)))
            o2, r, done, _ = env.step(a)  # XXX
            rtot += r
            if done:
                break
        rs[t] = rtot
    return rs.mean()
