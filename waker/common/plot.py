
import argparse
import collections
import functools
import itertools
import json
import multiprocessing as mp
import os
import pathlib
import re
import subprocess
import warnings

os.environ['NO_AT_BRIDGE'] = '1'  # Hide X org false warning.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

np.set_string_function(lambda x: f'<np.array shape={x.shape} dtype={x.dtype}>')

Run = collections.namedtuple('Run', 'task method seed xs ys')

PALETTES = dict(
    discrete=(
        '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
        '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
    ),
    contrast=(
        '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
        '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    ),
    gradient=(
        '#fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d',
        '#46327e', '#440154',
    ),
    baselines=(
        '#222222', '#666666', '#aaaaaa', '#cccccc',
    ),
)

LEGEND = dict(
    fontsize='medium', numpoints=1, labelspacing=0, columnspacing=1.2,
    handlelength=1.5, handletextpad=0.5, loc='lower center')

DEFAULT_BASELINES = ['d4pg', 'rainbow_sticky', 'human_gamer', 'impala']


def find_keys(args):
  filenames = []
  for indir in args.indir:
    task = next(indir.iterdir())  # First only.
    for method in task.iterdir():
      seed = next(indir.iterdir())  # First only.
      filenames += list(seed.glob('**/*.jsonl'))
  keys = set()
  for filename in filenames:
    keys |= set(load_jsonl(filename).columns)
  print(f'Keys      ({len(keys)}):', ', '.join(keys), flush=True)


def load_runs(args):
  total, toload = [], []
  for indir in args.indir:
    filenames = list(indir.glob('**/*.jsonl'))
    total += filenames
    for filename in filenames:
      task, method, seed = filename.relative_to(indir).parts[:-1]
      if not any(p.search(task) for p in args.tasks):
        continue
      if not any(p.search(method) for p in args.methods):
        continue
      toload.append((filename, indir))
  print(f'Loading {len(toload)} of {len(total)} runs...')
  jobs = [functools.partial(load_run, f, i, args) for f, i in toload]
  # Disable async data loading: