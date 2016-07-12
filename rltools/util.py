from __future__ import print_function

import errno
import os
import timeit

import h5py
import numpy as np
from colorama import Fore, Style



class Timer(object):
    def __enter__(self):
        self.t_start = timeit.default_timer()
        return self

    def __exit__(self, _1, _2, _3):
        self.t_end = timeit.default_timer()
        self.dt = self.t_end - self.t_start


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise



def split_h5_name(fullpath, sep='/'):
    """
    From h5ls.c:
     * Example: ../dir1/foo/bar/baz
     *          \_________/\______/
     *             file       obj
     *
    """
    sep_inds = [i for i, c in enumerate(fullpath) if c == sep]
    for sep_idx in sep_inds:
        filename, objname = fullpath[:sep_idx], fullpath[sep_idx:]
        if not filename: continue
        # Try to open the file. If it fails, try the next separation point.
        try: h5py.File(filename, 'r').close()
        except IOError: continue
        # It worked!
        return filename, objname
    raise IOError('Could not open HDF5 file/object {}'.format(fullpath))


def discount(r_N_T_D, gamma):
    '''
    Computes Q values from rewards.
    q_N_T_D[i,t,:] == r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + gamma^2*r_N_T_D[i,t+2,:] + ...
    '''
    assert r_N_T_D.ndim == 2 or r_N_T_D.ndim == 3
    input_ndim = r_N_T_D.ndim
    if r_N_T_D.ndim == 2: r_N_T_D = r_N_T_D[...,None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None,:,None]
    q_N_T_D = np.cumsum(discounted_N_T_D[:,::-1,:], axis=1)[:,::-1,:] # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None,:,None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:,-1,:], r_N_T_D[:,-1,:])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:,:,0]
    return q_N_T_D


def standardized(a):
    out = a.copy()
    out -= a.mean()
    out /= a.std() + 1e-8
    return out

def safezip(*ls):
    assert all(len(l) == len(ls[0]) for l in ls)
    return zip(*ls)

def maxnorm(a):
    return np.abs(a).max()

class Color(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def header(s): print(Color.HEADER + '{}'.format(s.encode('utf-8')) + Color.ENDC)
def warn(s): print(Color.WARNING + '{}'.format(s.encode('utf-8')) + Color.ENDC)
def failure(s): print(Color.FAIL + '{}'.format(s.encode('utf-8')) + Color.ENDC)
def ok(s): print(Color.OKBLUE + '{}'.format(s.encode('utf-8')) + Color.ENDC)
