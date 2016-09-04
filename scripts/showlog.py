#!/usr/bin/env python
#
# File: showlog.py
#
# Created: Monday, July 11 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--fields', type=str, default='ret,avglen,ent,kl,vf_r2,ttotal')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--plotfile', type=str, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    args = parser.parse_args()

    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fields = args.fields.split(',')

    # Load logs from all files
    fname2log = {}
    for fname in args.logfiles:
        if ':' in fname:
            os.system('rsync -avrz {} /tmp'.format(fname))
            fname = os.path.join('/tmp', os.path.basename(fname))
        with pd.HDFStore(fname, 'r') as f:
            assert fname not in fname2log
            df = f['log']
            df.set_index('iter', inplace=True)
            fname2log[fname] = df.loc[:args.range_end, fields]

    # Print
    if not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-colorblind')

    ax = None
    for fname, df in fname2log.items():
        with pd.option_context('display.max_rows', 9999):
            print(fname)
            print(df[-1:])

        if 'vf_r2' in df.keys():
            df['vf_r2'] = np.maximum(0, df['vf_r2'])

        if not args.noplot:
            if ax is None:
                ax = df.plot(subplots=True, title=','.join(args.logfiles))
            else:
                df.plot(subplots=True, title=','.join(args.logfiles), ax=ax, legend=False)
    if args.plotfile is not None:
        plt.savefig(args.plotfile, transparent=True, bbox_inches='tight', dpi=300)
    elif not args.noplot:
        plt.show()


if __name__ == '__main__':
    main()
