"""Eyes-open vs eyes-closed resting state (PhysioNet EEGMMIDB runs 1 and 2, 109 subjects).

States are fitted without condition labels on pooled data (100 random time points per subject and
condition); every sample is then assigned, and each subject's fractional occupancy (FO) in each
condition is used to decode eyes open vs closed with leave-one-subject-out logistic regression.
Baseline: EEG microstates (2-20 Hz, average reference, polarity-invariant K-means on GFP-peak maps).

Run:  python eeg_eoec.py [path/to/eegmmidb_raw]   -> ../results/eoec.npz, eoec_summary.json
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
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import iplgeom as ig
from edf import read_edf

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'eegmmidb_raw')
RES = os.path.join(HERE, '..', 'results')
CACHE = os.path.join(RES, 'eoec_cache')
FS, BAND, ORDER, DECIM, TRIM, T_RAW = 160, (8.0, 13.0), 3, 4, 40, 9600
N_PER, N_INIT, KS = 100, 10, list(range(2, 9))
METHODS = ig.COSINE_METHODS + ['microstates']

Z = None        # (2, S, T, p) alpha phase vectors, condition 0 = eyes open, 1 = eyes closed
MAPS = None     # list over (cond, subj) of normalized GFP-peak maps (n_peaks, p)
TOPO = None     # (2, S, T_raw, p) normalized 2-20 Hz topographies for microstate backfitting
FIT = None      # (2, S, N_PER) fit indices


def microstate_method():
    # polarity-invariant (diametrical) K-means of average-referenced, unit-norm topographies
    m = ig.METHODS['leida']
    return ig.Method('microstates', 'Microstates', lambda x: x, m.dist, m.update)


def load():
    global Z, MAPS, TOPO, FIT
    b_ms, a_ms = butter(ORDER, np.array([2.0, 20.0]) / (FS / 2), btype='band')
    Z, TOPO, MAPS = [], [], []
    for run in (1, 2):
        zr, tr = [], []
        for s in range(1, 110):
            _, fs, X = read_edf(os.path.join(RAW, f'S{s:03d}R{run:02d}.edf'))
            X = X[:, :T_RAW]
            zr.append(ig.phase_vectors(ig.instantaneous_phase(X, FS, BAND, ORDER, DECIM, TRIM)))
            Y = filtfilt(b_ms, a_ms, X, axis=1)
            Y = Y - Y.mean(0, keepdims=True)                 # average reference
            gfp = Y.std(0)
            peaks, _ = find_peaks(gfp)
            Yn = (Y / np.maximum(np.linalg.norm(Y, axis=0), 1e-12)).T
            MAPS.append(Yn[peaks])
            tr.append(Yn.astype(np.float32))
        Z.append(np.stack(zr))
        TOPO.append(np.stack(tr))
    Z, TOPO = np.stack(Z), np.stack(TOPO)
    rng = np.random.default_rng(0)
    FIT = np.stack([np.stack([np.sort(rng.choice(Z.shape[2], N_PER, replace=False)) for _ in range(Z.shape[1])])
                    for _ in range(2)])


def fit_data(name):
    S = Z.shape[1]
    if name == 'microstates':
        rng = np.random.default_rng(1)
        return np.concatenate([MAPS[c * S + s][rng.choice(len(MAPS[c * S + s]), N_PER, replace=len(MAPS[c * S + s]) < N_PER)]
                               for c in range(2) for s in range(S)])
    return np.concatenate([Z[c, s, FIT[c, s]] for c in range(2) for s in range(S)])


def task(args):
    name, K = args
    path = os.path.join(CACHE, f'{name}_K{K}.npz')
    if os.path.exists(path):
        r = np.load(path)
        return name, K, r['fo'], r['C'], r['lab_fit']
    m = microstate_method() if name == 'microstates' else ig.METHODS[name]
    X = fit_data(name)
    Xp = X if name == 'microstates' else m.prepare(X)
    lab_fit, C, _ = ig.fit(m, Xp, K, n_init=N_INIT, seed=K)
    S = Z.shape[1]
    fo = np.zeros((2, S, K))
    for c in range(2):
        for s in range(S):
            if name == 'microstates':
                lab = m.dist(TOPO[c, s].astype(np.float64), C).argmin(1)
            else:
                lab = ig.assign(m, Z[c, s], C)
            fo[c, s] = np.bincount(lab, minlength=K) / len(lab)
    np.savez(path, fo=fo, C=C, lab_fit=lab_fit)
    return name, K, fo, C, lab_fit


def decode(fo):
    """Leave-one-subject-out logistic regression of eyes open (0) vs closed (1) from FO."""
    S, K = fo.shape[1], fo.shape[2]
    X = np.concatenate([fo[0], fo[1]])
    y = np.r_[np.zeros(S), np.ones(S)]
    groups = np.r_[np.arange(S), np.arange(S)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    pred = cross_val_predict(clf, X, y, groups=groups, cv=LeaveOneGroupOut())
    return float(np.mean(pred == y))


def paired_effect(fo):
    d = fo[1] - fo[0]                                      # eyes closed minus eyes open, per subject
    return d.mean(0) / np.maximum(d.std(0, ddof=1), 1e-12)


def main():
    t0 = time.time()
    load()
    os.makedirs(CACHE, exist_ok=True)
    lam_eo = ig.ipl_eig(Z[0].reshape(-1, Z.shape[-1]))[1][:, 0] / Z.shape[-1]
    lam_ec = ig.ipl_eig(Z[1].reshape(-1, Z.shape[-1]))[1][:, 0] / Z.shape[-1]
    print(f'loaded ({time.time() - t0:.0f} s); lambda1/p EO {lam_eo.mean():.3f}, EC {lam_ec.mean():.3f}', flush=True)
    order = sorted(METHODS, key=lambda n: (n not in ('eida', 'sine'), n))
    with Pool(int(os.environ.get('N_PROC', 42))) as pool:
        res = pool.map(task, [(n, K) for n in order for K in KS[::-1]], chunksize=1)
    out, summary = {'ks': np.array(KS), 'methods': np.array(METHODS)}, {'methods': {}}
    summary['lambda1_over_p'] = {'eyes_open': float(lam_eo.mean()), 'eyes_closed': float(lam_ec.mean())}
    zf = fit_data('bw')
    for name, K, fo, C, lab_fit in res:
        acc = decode(fo)
        eff = paired_effect(fo)
        occ = fo.mean((0, 1))
        rank = np.argsort(-occ)
        entry = {'accuracy': acc, 'occupancy': occ[rank].round(4).tolist(),
                 'fo_eyes_open': fo[0].mean(0)[rank].round(4).tolist(),
                 'fo_eyes_closed': fo[1].mean(0)[rank].round(4).tolist(),
                 'paired_d': eff[rank].round(3).tolist()}
        summary['methods'].setdefault(name, {})[K] = entry
        out[f'{name}_K{K}_fo'] = fo[:, :, rank]
        if name != 'microstates':
            rank_inv = np.empty(K, int)
            rank_inv[rank] = np.arange(K)
            lf = rank_inv[lab_fit]
            out[f'{name}_K{K}_maps'] = np.stack([ig.bw_barycentre(zf[lf == k]) for k in range(K)])
        else:
            out[f'{name}_K{K}_maps'] = C[rank]
    np.savez_compressed(os.path.join(RES, 'eoec.npz'), **out)
    json.dump(summary, open(os.path.join(RES, 'eoec_summary.json'), 'w'), indent=1)
    print(f'saved ({time.time() - t0:.0f} s)')
    for name in METHODS:
        print(f'{name:12s} EO/EC accuracy K=2..8:', [round(summary["methods"][name][K]["accuracy"], 2) for K in KS])


if __name__ == '__main__':
    main()
