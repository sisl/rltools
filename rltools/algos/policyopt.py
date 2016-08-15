from __future__ import absolute_import, print_function
import numpy as np

from rltools import util
from rltools.algos import RLAlgorithm
from rltools.samplers.serial import SimpleSampler
from rltools.samplers.parallel import ParallelSampler


class SamplingPolicyOptimizer(RLAlgorithm):

    def __init__(self, env, policy, baseline, step_func, obsfeat_fn=lambda obs: obs, discount=0.99,
                 gae_lambda=1, n_iter=500, start_iter=0, center_adv=True, positive_adv=False,
                 store_paths=False, whole_paths=True, sampler_cls=None,
                 sampler_args=dict(max_traj_len=200, n_timesteps=6400, adaptive=False,
                                   n_timesteps_min=1600, n_timesteps_max=12800, timestep_rate=40,
                                   enable_rewnorm=True), **kwargs):
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.step_func = step_func
        self.obsfeat_fn = obsfeat_fn
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_iter = n_iter
        self.start_iter = start_iter
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths  # TODO
        self.whole_paths = whole_paths  # TODO
        if sampler_cls is None:
            sampler_cls = SimpleSampler
        self.sampler = sampler_cls(self, **sampler_args)
        self.total_time = 0.0

    def train(self, sess, log, save_freq):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0 or itr % self.n_iter:
                log.write_snapshot(sess, self.policy, itr)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            # Sample trajs using current policy
            with util.Timer() as t_sample:
                if itr == 0:
                    # extra batch to init std
                    trajbatch0, _ = self.sampler.sample(sess, itr)
                    self.policy.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                    self.baseline.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                    self.sampler.rewnorm.update(sess, trajbatch0.r.stacked[:, None])
                trajbatch, sample_info_fields = self.sampler.sample(sess, itr)

            # Compute baseline
            with util.Timer() as t_base:
                trajbatch_vals, base_info_fields = self.sampler.process(sess, itr, trajbatch,
                                                                        self.discount,
                                                                        self.gae_lambda,
                                                                        self.baseline)

            # Take the policy grad step
            with util.Timer() as t_step:
                params0_P = self.policy.get_params(sess)
                step_print_fields = self.step_func(sess, self.policy, trajbatch,
                                                   trajbatch_vals['advantage'])
                self.policy.update_obsnorm(sess, trajbatch.obsfeat.stacked)
                self.sampler.rewnorm.update(sess, trajbatch.r.stacked[:, None])
        # LOG
        self.total_time += t_all.dt

        fields = [
            ('iter', itr, int)
        ] + sample_info_fields + [
            ('vf_r2', trajbatch_vals['v_r'], float),
            ('tdv_r2', trajbatch_vals['tv_r'], float),
            ('ent', self.policy._compute_actiondist_entropy(trajbatch.adist.stacked).mean(), float
            ),  # entropy of action distribution
            ('dx', util.maxnorm(params0_P - self.policy.get_params(sess)), float
            )  # max parameter different from last iteration
        ] + base_info_fields + step_print_fields + [
            ('tsamp', t_sample.dt, float),  # Time for sampling
            ('tbase', t_base.dt, float),  # Time for advantage/baseline computation
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields


def TRPO(max_kl, subsample_hvp_frac=.1, damping=1e-2, grad_stop_tol=1e-6, max_cg_iter=10,
         enable_bt=True):

    def trpo_step(sess, policy, trajbatch, advantages):
        # standardize advantage
        advstacked_N = util.standardized(advantages.stacked)

        # Compute objective, KL divergence and gradietns at init point
        feed = (trajbatch.obsfeat.stacked, trajbatch.a.stacked, trajbatch.adist.stacked,
                advstacked_N)

        step_info = policy._ngstep(sess, feed, max_kl=max_kl, damping=damping,
                                   subsample_hvp_frac=subsample_hvp_frac,
                                   grad_stop_tol=grad_stop_tol)
        return [
            ('dl', step_info.obj1 - step_info.obj0, float),  # Improvement in objective
            ('kl', step_info.kl1, float),  # kl cost
            ('gnorm', step_info.gnorm, float),  # gradient norm
            ('bt', step_info.bt, int),  # backtracking steps
        ]

    return trpo_step


class ConcurrentPolicyOptimizer(RLAlgorithm):

    def __init__(self, env, policies, baselines, step_func, target_policy, weights, interp_alpha,
                 discount, gae_lambda, n_iter, start_iter=0, sampler_cls=None, sampler_args=None,
                 **kwargs):
        self.env = env
        self.policies = policies
        self.baselines = baselines
        self.step_func = step_func
        self.target_policy = target_policy
        self.weights = weights
        self.interp_alpha = interp_alpha
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.n_iter = n_iter
        self.start_iter = start_iter
        if sampler_cls is None:
            sampler_cls = ParallelSampler
        self.sampler = sampler_cls(self, **sampler_args)
        self.total_time = 0.0

    def train(self, sess, log, blend_freq, save_freq):
        for itr in range(self.start_iter, self.n_iter):
            iter_info = self.step(sess, itr)
            log.write(iter_info, print_header=itr % 20 == 0)
            if itr % save_freq == 0 or itr % self.n_iter:
                for policy in self.policies:
                    log.write_snapshot(sess, policy, itr)

            if itr % blend_freq == 0 or itr % self.n_iter:
                assert self.target_policy is not None
                params_P_ag = [policy.get_params(sess) for policy in self.policies]
                weightparams_P = np.sum([w * p for w, p in util.safezip(self.weights, params_P_ag)])
                blendparams_P = self.interp_alpha * self.target_policy.get_params(sess) + (
                    1 - self.interp_alpha) * weightparams_P
                self.target_policy.set_params(sess, blendparams_P)

    def step(self, sess, itr):
        with util.Timer() as t_all:
            with util.Timer() as t_sample:
                if itr == 0:
                    # extra batch to init std
                    trajbatchlist0, _ = self.sampler.sample(sess, itr)
                    for policy, baseline, trajbatch0 in util.safezip(self.policies, self.baselines,
                                                                     trajbatchlist0):
                        policy.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                        baseline.update_obsnorm(sess, trajbatch0.obsfeat.stacked)
                        self.sampler.rewnorm.update(sess, trajbatch0.r.stacked[:, None])
                trajbatchlist, sampler_info_fields = self.sampler.sample(sess, itr)

            # Baseline
            with util.Timer() as t_base:
                trajbatch_vals_list, base_info_fields_list = [], []
                for agid, trajbatch in enumerate(trajbatchlist):
                    trajbatch_vals, base_info_fields = self.sampler.process(sess, itr, trajbatch,
                                                                            self.discount,
                                                                            self.gae_lambda,
                                                                            self.baselines[agid])
                    trajbatch_vals_list.append(trajbatch_vals)
                    base_info_fields_list += base_info_fields

            # Take policy steps
            with util.Timer() as t_step:
                step_print_fields_list = []
                params0_P_list = []
                for agid, policy in enumerate(self.policies):
                    params0_P = policy.get_params(sess)
                    params0_P_list.append(params0_P)
                    step_print_fields = self.step_func(sess, policy, trajbatchlist[agid],
                                                       trajbatch_vals_list[agid]['advantage'])
                    step_print_fields_list += step_print_fields
                    policy.update_obsnorm(sess, trajbatchlist[agid].obsfeat.stacked)
                    self.sampler.rewnorm.update(sess, trajbatchlist[agid].r.stacked[:, None])

                # LOG
        self.total_time += t_all.dt

        infos = []
        for agid in range(len(self.policies)):
            infos += [
                ('vf_r2_{}'.format(agid), trajbatch_vals_list[agid]['v_r'], float),
                ('tdv_r2_{}'.format(agid), trajbatch_vals_list[agid]['tv_r'], float),
                ('ent_{}'.format(agid), self.policies[agid]._compute_actiondist_entropy(
                    trajbatchlist[agid].adist.stacked).mean(), float),
                ('dx_{}'.format(agid),
                 util.maxnorm(params0_P_list[agid] - self.policies[agid].get_params(sess)), float)
            ]
        fields = [
            ('iter', itr, int)
        ] + sampler_info_fields + infos + base_info_fields_list + step_print_fields_list + [
            ('tsamp', t_sample.dt, float),  # Time for sampling
            ('tbase', t_base.dt, float),  # Time for advantage/baseline computation
            ('tstep', t_step.dt, float),
            ('ttotal', self.total_time, float)
        ]
        return fields
