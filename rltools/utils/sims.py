

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

