"""Attack-free (self-supervised) APC training data: clean point clouds paired with
synthetic corruptions. Motivated by the mechanism analysis — APC learns an
input-canonicalization (applied to clean and attacked inputs alike, counter-offset
orthogonal to the true attack direction), so adversarial pairs may be unnecessary.

Corruption menu, reduced to 4 ops by the 2026-10-16 ablation (see note on _patch_drop):
  point shifting (jitter / sparse shift), object relocation (shift),
  point dropping (contiguous patch drop).
Still in OPS but off the default menu: outlier, dropout, bump.

Configurable via args (all optional):
  --synth_ops      subset of op names to build the menu from (default: DEFAULT_MENU);
                   "op@w" runs that op at w x severity, e.g. "patchdrop@2"
  --synth_nops     comma list to sample #ops per example from (default '1,1,2')
  --synth_severity global magnitude multiplier s (default 1.0)
"""
import random
from collections import defaultdict

import numpy as np
from torch.utils.data import Dataset

from util.pointnet_utils import random_sample_points_np, normalize_points_np
from util.augmentation import rotate_point_cloud, jitter_point_cloud


def _jitter(pc, s=1.0):
    sigma = np.random.uniform(0.01, 0.04) * s
    return pc + np.random.normal(0., sigma, pc.shape)


def _sparse_shift(pc, s=1.0):
    n = len(pc)
    k = int(n * np.random.uniform(0.1, 0.3))
    idx = np.random.choice(n, k, replace=False)
    out = pc.copy()
    out[idx] += np.random.normal(0., np.random.uniform(0.05, 0.15) * s, (k, 3))
    return out


def _outliers(pc, s=1.0):
    n = len(pc)
    k = int(n * min(np.random.uniform(0.02, 0.10) * s, 0.5))
    k = max(k, 1)
    idx = np.random.choice(n, k, replace=False)
    out = pc.copy()
    scale = np.abs(pc).max()
    out[idx] = np.random.uniform(-1., 1., (k, 3)) * scale
    return out


def _dropout(pc, s=1.0):
    n = len(pc)
    keep = int(n * (1. - min(np.random.uniform(0.1, 0.3) * s, 0.7)))
    idx = np.random.choice(n, keep, replace=False)
    return random_sample_points_np(pc[idx], n)


def _local_bump(pc, s=1.0):
    # TODO: make this more shape-aware (e.g. bump along normal direction, or at least in-plane)
    # TODO: multiple bumps per example (maybe 1-3, each with random radius and magnitude)
    c = pc[np.random.randint(len(pc))]
    r = np.random.uniform(0.1, 0.3)
    d2 = ((pc - c) ** 2).sum(1)
    w = np.exp(-d2 / (r ** 2))
    direction = np.random.randn(3)
    direction /= (np.linalg.norm(direction) + 1e-9)
    mag = np.random.uniform(0.05, 0.2) * s
    return pc + w[:, None] * direction[None] * mag


def _patch_drop(pc, s=1.0):
    """Remove a few contiguous local regions (mimics saliency-guided point dropping
    better than uniform-random dropout), then bootstrap-resample to N.

    2026-10-16: strength doubled (was U(0.1, 0.25)) so this single op covers what the
    two-entry menu ['patchdrop', 'patchdrop@2'] used to. 'patchdrop' now means the old
    'patchdrop@2'; runs logged before this date used the 1x version under the same name
    and will not reproduce as-is."""
    n = len(pc)
    remove = int(n * min(np.random.uniform(0.2, 0.5) * s, 0.6))
    seeds = pc[np.random.choice(n, np.random.randint(2, 6), replace=False)]
    d = np.linalg.norm(pc[:, None] - seeds[None], axis=2).min(1)
    drop_idx = np.argsort(d)[:remove]
    keep = np.setdiff1d(np.arange(n), drop_idx)
    return random_sample_points_np(pc[keep], n)


def _shift(pc, s=1.0):
    """global translation of the whole object (relocation)"""
    d = np.random.randn(3)
    d /= (np.linalg.norm(d) + 1e-9)
    return pc + d * np.random.uniform(0., 1.2) * s


def _identity(pc, s=1.0):
    """no corruption (clean->clean pairs); ablation control only"""
    return pc


OPS = {
    'identity': _identity,
    'jitter': _jitter,
    'sparse': _sparse_shift,
    'outlier': _outliers,
    'dropout': _dropout,
    'bump': _local_bump,
    'patchdrop': _patch_drop,
    'shift': _shift,
}
# fallback when args.synth_ops is unset; mirrors af_main.py's --synth_ops default
DEFAULT_MENU = ['jitter', 'sparse', 'shift', 'patchdrop']


def corrupt(pc, menu=None, nops_pool=(1, 1, 2), severity=1.0):
    menu = DEFAULT_MENU if menu is None else menu
    ops = random.sample(menu, k=min(random.choice(nops_pool), len(menu)))
    out = pc
    for name in ops:
        name, _, w = name.partition('@')     # "op@w" -> this op runs at w x severity
        out = OPS[name](out, severity * (float(w) if w else 1.))
    return out.astype(np.float32)


class SynthCorruptSet(Dataset):
    """Returns (corrupted_pc, label, clean_pc) — same interface as the APCHybrid sets.

    dataset='sonn' (default) loads ScanObjectNN `sonn_type`; dataset='mn40' loads
    ModelNet40 (normalized, mirroring dataset.ModelNet40)."""
    def __init__(self, args, sonn_type, num_points, partition, frac=1.0,
                 augmentation=None, dataset='sonn'):
        self.dataset = dataset
        if dataset == 'mn40':
            from dataset.ModelNet40 import load_data as load_mn40
            root = f'data/modelnet40_normal_resampled/modelnet40_{partition}_1024pts.dat'
            data, label = load_mn40(root, partition=partition)
        else:
            from dataset.ScanObjectNN import load_data
            data, label = load_data(sonn_type, split=partition)

        if frac < 1.0:
            datum = defaultdict(list)
            for d, l in zip(data, label):
                datum[l.item()].append(d)
            ds, ls = [], []
            for i in range(len(datum)):
                d = datum[i]
                idx = random.sample(range(len(d)), round(len(d) * frac))
                ds.extend([d[j] for j in idx])
                ls.extend([i] * len(idx))
            data, label = np.stack(ds), np.array(ls)

        self.data, self.label = data, label
        self.num_points = num_points
        self.partition = partition
        self.augmentation = (partition == 'train') if augmentation is None else augmentation

        # corruption config
        menu = getattr(args, 'synth_ops', None)
        if menu:
            for m in menu:
                assert m.split('@')[0] in OPS, f'unknown synth op {m}; choose from {list(OPS)}'
        self.menu = list(menu) if menu else None
        nops = getattr(args, 'synth_nops', '1,1,2') or '1,1,2'
        self.nops_pool = tuple(int(x) for x in str(nops).split(','))
        sev = str(getattr(args, 'synth_severity', 1.0) or 1.0)
        if '-' in sev:      # per-sample random severity range, e.g. "0.5-1.5"
            lo, hi = sev.split('-')
            self.severity = (float(lo), float(hi))
        else:
            self.severity = (float(sev), float(sev))

    def __getitem__(self, item):
        pc = self.data[item][:, :3]
        if self.partition == 'test':
            pc = pc[:self.num_points]
        else:
            pc = random_sample_points_np(pc, self.num_points)
        if self.dataset == 'mn40':
            pc = normalize_points_np(pc)
        if self.augmentation:
            pc = rotate_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        pc = pc.astype(np.float32)
        s = np.random.uniform(*self.severity) if self.severity[0] != self.severity[1] else self.severity[0]
        corrupted_pc = corrupt(pc, self.menu, self.nops_pool, s)
        return corrupted_pc, self.label[item], pc

    def __len__(self):
        return self.data.shape[0]
