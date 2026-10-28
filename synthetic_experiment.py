"""Synthetic benchmark on the 64-channel EEG montage.

Scenarios
  main      : 4 states = {anterior-posterior, left-right} phase gradient x {small, large} lag amplitude
              (states differ both in spatial pattern and in degree of synchrony)  -> Fig. 1
  shape     : 4 states that differ only in spatial pattern (AP, LR, diagonal, radial), large amplitude
  conjugate : 2 mirror-image travelling waves, theta and -theta (forward vs backward propagation)
Every scenario sweeps the i.i.d. phase-noise s.d. sigma; each sample also gets a uniformly random
global phase. All methods are fit to the same samples with K-means++ initialization and restarts.

Run:  python synthetic_experiment.py   -> ../results/synthetic.npz, ../results/synthetic_summary.json
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import json
import sys
import time
from multiprocessing import Pool
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari

sys.path.insert(0, os.path.dirname(__file__))
import iplgeom as ig
from eeg_layout import XY

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
METHODS = ig.COSINE_METHODS + ['complex', 'bw_medoids']
SIGMAS = np.round(np.linspace(0.05, 1.0, 12), 3)
N_REPS = 20
N_PER = 250
N_INIT = 5
SMALL, LARGE = 0.5, 1.5          # peak-to-peak lag across the scalp (radians)


def unit_gradient(v):
    return (v - v.mean()) / np.ptp(v)


G_AP, G_LR = unit_gradient(XY[:, 1]), unit_gradient(XY[:, 0])
G_DIAG = unit_gradient(XY[:, 0] + XY[:, 1])
G_RAD = unit_gradient(np.linalg.norm(XY, axis=1))

SCENARIOS = {
    'main': np.stack([SMALL * G_AP, LARGE * G_AP, SMALL * G_LR, LARGE * G_LR]),
    'shape': np.stack([LARGE * G_AP, LARGE * G_LR, LARGE * G_DIAG, LARGE * G_RAD]),
    'conjugate': np.stack([LARGE * G_AP, -LARGE * G_AP]),
}


def bw_kmedoids(z, K, n_init, rng):
    """K-medoids (alternating/Voronoi iteration) with the BW distance, ++ seeding."""
    D = ig.bw_sqdist(z, z)
    n = len(z)
    best = (np.inf, None)
    for _ in range(n_init):
        m = [int(rng.integers(n))]
        dmin = D[:, m[0]].copy()
        for _ in range(K - 1):
            m.append(int(rng.choice(n, p=dmin / dmin.sum())))
            dmin = np.minimum(dmin, D[:, m[-1]])
        m = np.array(m)
        for _ in range(200):
            lab = D[:, m].argmin(1)
            new = m.copy()
            for k in range(K):
                idx = np.flatnonzero(lab == k)
                if len(idx):
                    new[k] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
            if np.array_equal(new, m):
                break
            m = new
        lab = D[:, m].argmin(1)
        cost = D[np.arange(n), m[lab]].sum()
        if cost < best[0]:
            best = (cost, lab)
    return best[1]


def run_one(args):
    scenario, si, rep = args
    templates = SCENARIOS[scenario]
    seed = 1000 * rep + 37 * si + {'main': 0, 'shape': 1, 'conjugate': 2}[scenario]
    rng = np.random.default_rng(seed)
    z, y = ig.simulate(templates, N_PER, SIGMAS[si], rng)
    out = {}
    for name in METHODS:
        if name == 'bw_medoids' and scenario != 'main':
            continue
        t0 = time.perf_counter()
        if name == 'bw_medoids':
            lab = bw_kmedoids(z, len(templates), N_INIT, rng)
        else:
            m = ig.METHODS[name]
            lab = ig.fit(m, m.prepare(z), len(templates), n_init=N_INIT, seed=seed)[0]
        out[name] = (nmi(y, lab), ari(y, lab), time.perf_counter() - t0)
    lam1 = ig.ipl_eig(z)[1][:, 0].mean() / z.shape[1]
    return scenario, si, rep, out, lam1


if __name__ == '__main__':
    tasks = [(s, si, r) for s in SCENARIOS for si in range(len(SIGMAS)) for r in range(N_REPS)]
    t0 = time.time()
    with Pool(int(os.environ.get('N_PROC', 48))) as pool:
        results = pool.map(run_one, tasks, chunksize=1)
    print(f'{len(tasks)} runs in {time.time() - t0:.0f} s')
    shape = (len(SIGMAS), N_REPS)
    store = {}
    for scen in SCENARIOS:
        for name in METHODS:
            for j, metric in enumerate(['nmi', 'ari', 'time']):
                store[f'{scen}_{name}_{metric}'] = np.full(shape, np.nan)
        store[f'{scen}_lambda1'] = np.full(shape, np.nan)
    for scen, si, rep, out, lam1 in results:
        for name, vals in out.items():
            for j, metric in enumerate(['nmi', 'ari', 'time']):
                store[f'{scen}_{name}_{metric}'][si, rep] = vals[j]
        store[f'{scen}_lambda1'][si, rep] = lam1
    os.makedirs(RES, exist_ok=True)
    np.savez(os.path.join(RES, 'synthetic.npz'), sigmas=SIGMAS, methods=np.array(METHODS),
             small=SMALL, large=LARGE, n_per=N_PER, n_reps=N_REPS, **store,
             **{f'templates_{k}': v for k, v in SCENARIOS.items()})
    summary = {}
    for scen in SCENARIOS:
        summary[scen] = {'lambda1_over_p': np.nanmean(store[f'{scen}_lambda1'], 1).round(3).tolist()}
        print(f'\n[{scen}] mean NMI; sigma =', SIGMAS.tolist())
        print('  lambda1/p:', summary[scen]['lambda1_over_p'])
        for name in METHODS:
            v = store[f'{scen}_{name}_nmi']
            if np.all(np.isnan(v)):
                continue
            summary[scen][name] = {'nmi_mean': np.nanmean(v, 1).round(3).tolist(),
                                   'nmi_sd': np.nanstd(v, 1).round(3).tolist(),
                                   'time_mean_s': float(np.nanmean(store[f'{scen}_{name}_time']))}
            print(f'  {name:12s}', ' '.join(f'{x:5.2f}' for x in np.nanmean(v, 1)))
    json.dump(summary, open(os.path.join(RES, 'synthetic_summary.json'), 'w'), indent=1)
