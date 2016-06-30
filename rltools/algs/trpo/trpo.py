from trpo_utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import prettytensor as pt


class TRPOSolver(object):

    def __init__(self, env, config=None, policy_net=None):
        self.env = env

        if config is None: config = {}
        config.setdefault('train_iterations', 100)
        config.setdefault('max_pathlength', 100)
        config.setdefault('timesteps_per_batch', 1000)
        config.setdefault('eval_trajectories', 10)
        config.setdefault('max_kl', 0.01)
        config.setdefault('gamma', 0.95)
        config.setdefault('explained_variance_thresh', 0.95)

        self.config = config

        self.session = tf.Session()
        self.end_count = 0
        self.train = True

        dtype = tf.float32

        self.obs = obs = tf.placeholder(dtype, shape=[None,] + list(env.observation_space.shape), name="obs")

        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")  
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")  
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.n], name="oldaction_dist")

        # Create neural network.
        if policy_net is None:
            action_dist_n, _ = (pt.wrap(self.obs).
                                fully_connected(32, activation_fn=tf.nn.tanh).
                                fully_connected(32, activation_fn=tf.nn.tanh).
                                softmax_classifier(env.action_space.n))
        else:
            action_dist_n = policy_net
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        p_n = slice_2d(action_dist_n, tf.range(0, N), action)
        oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
        ratio_n = p_n / oldp_n
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        var_list = tf.trainable_variables()
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (action_dist_n + eps))) / Nf
        ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + eps)) / Nf

        self.train_stats = {}
        self.train_stats["Total number of episodes"] = []
        self.train_stats["Average sum of rewards per episode"] = []
        self.train_stats["Eval Reward"] = []
        self.train_stats["Entropy"] = []
        self.train_stats["Baseline explained"] = []
        self.train_stats["Time elapsed"] = []
        self.train_stats["KL between old and new distribution"] = []
        self.train_stats["Surrogate loss"] = []

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        shapes = map(var_shape, var_list)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs})
        if self.train:
            action = int(cat_sample(action_dist_n)[0])
        else:
            action = int(np.argmax(action_dist_n))
        return action, action_dist_n

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        thresh_it = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                self.config["max_pathlength"],
                self.config["timesteps_per_batch"])

            if not paths:
                # if goal not reached in trajectory
                print "Paths empty"
                self.train = True
                continue

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config["gamma"])
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                self.advant: advant_n,
                    self.oldaction_dist: action_dist_n}


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                #self.train = True
                if self.end_count > self.config["train_iterations"]:
                    break
            if self.train: 
                self.vf.fit(paths)
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed)

                g = self.session.run(self.pg, feed_dict=feed)
                stepdir = conjugate_gradient(fisher_vector_product, -g)
                shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / config["max_kl"])
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.sff(th)
                    return self.session.run(self.losses[0], feed_dict=feed)
                theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                theta = thprev + fullstep
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config["max_kl"]:
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Eval Reward"] = eval_policy(self.env, self, self.config["eval_trajectories"], self.config["max_pathlength"])
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.iteritems():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                    self.train_stats[k].append(v)
                if entropy != entropy:
                    # not sure what to do here?
                    break
                    exit(-1)
                if exp > self.config["explained_variance_thresh"]:
                    self.train = False
                    break
            if i >= self.config["train_iterations"]:
                break
            i += 1
