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
        if not filename:
            continue
        # Try to open the file. If it fails, try the next separation point.
        try:
            h5py.File(filename, 'r').close()
        except IOError:
            continue
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
    if r_N_T_D.ndim == 2:
        r_N_T_D = r_N_T_D[..., None]

    discfactors_T = np.power(gamma, np.arange(r_N_T_D.shape[1]))
    discounted_N_T_D = r_N_T_D * discfactors_T[None, :, None]
    q_N_T_D = np.cumsum(
        discounted_N_T_D[:, ::-1, :],
        axis=1)[:, ::
                -1, :]  # this is equal to gamma**t * (r_N_T_D[i,t,:] + gamma*r_N_T_D[i,t+1,:] + ...)
    q_N_T_D /= discfactors_T[None, :, None]

    # Sanity check: Q values at last timestep should equal original rewards
    assert np.allclose(q_N_T_D[:, -1, :], r_N_T_D[:, -1, :])

    if input_ndim == 2:
        assert q_N_T_D.shape[-1] == 1
        return q_N_T_D[:, :, 0]
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


def gather(vals, idx):
    return vals[idx]


def lookup_last_idx(a, inds):
    """
    Looks up indices in a. e.g. a[[1, 2, 3]] = [a[1], a[2], a[3]]
    a is a d1 x d2 ... dn array
    inds is a d1 x d2 ... d(n-1) array of integers
    returns the array
    out[i_1,...,i_{n-1}] = a[i_1,...,i_{n-1}, inds[i_1,...,i_{n-1}]]
    """

    # Flatten the arrays
    ashape, indsshape = np.shape(a), np.shape(inds)
    aflat, indsflat = np.reshape(a, (-1,)), np.reshape(inds, (-1,))

    # Compute the indices corresponding to inds in the flattened array
    delta = gather(ashape, np.size(ashape) - 1)  # i.e. delta = ashape[-1],
    aflatinds = np.arange(0, stop=np.size(a), step=delta) + indsflat

    # Look up the desired elements in the flattened array, and reshape
    # to the original shape
    return np.reshape(gather(aflat, aflatinds), indsshape)


class Color(object):
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def header(s):
    print(Color.HEADER + '{}'.format(s) + Color.ENDC)


def warn(s):
    print(Color.WARNING + '{}'.format(s) + Color.ENDC)


def failure(s):
    print(Color.FAIL + '{}'.format(s) + Color.ENDC)


def ok(s):
    print(Color.OKBLUE + '{}'.format(s) + Color.ENDC)


class EzPickle(object):
    """Objects that are pickled and unpickled via their constructor
    arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
                ...

    When this object is unpickled, a new Dog will be constructed by passing the provided
    furcolor and tailkind into the constructor. However, philosophers are still not sure
    whether it is still the same dog.

    """

    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs

    def __getstate__(self):
        return {"_ezpickle_args": self._ezpickle_args, "_ezpickle_kwargs": self._ezpickle_kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)
