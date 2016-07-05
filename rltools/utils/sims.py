import numpy as np

def simulate(env, policy, maxsteps, render=False):
    obs = env.reset()
    rtot = 0.0
    for _ in xrange(maxsteps):
        if render: env.render()
        action, action_dist = policy.act(obs)
        obs, reward, done, _ = env.step(action)
        rtot += reward
        if done:
            break
    return rtot



def evaluate(env, policy, maxsteps, n_traj, render=False):
    obs = env.reset()
    rewards = np.zeros(n_traj)
    for i in xrange(n_traj):
        rtot = 0.0
        for _ in xrange(maxsteps):
            if render: env.render()
            action, action_dist = policy.act(obs)
            obs, reward, done, _ = env.step(action)
            rtot += reward
            if done:
                break
        rewardsi[i] = rtot
    return rewards

