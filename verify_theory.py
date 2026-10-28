"""Numerical verification of every identity used in the paper (Sec. 2).

Run:  python verify_theory.py   ->  prints a table and writes ../results/theory_checks.json
"""
import json
import os
import sys
import numpy as np
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(__file__))
import iplgeom as ig

rng = np.random.default_rng(2026)
p, n_trials = 64, 300
RES = os.path.join(os.path.dirname(__file__), '..', 'results')


def random_phases(n):
    # mixture of near-synchronous and dispersed patterns
    spread = rng.uniform(0.05, np.pi, (n, 1))
    return rng.uniform(-1, 1, (n, p)) * spread + rng.uniform(0, 2 * np.pi, (n, 1))


def bw_matrix(Pa, Pb):
    ra = np.real(sqrtm(Pa))
    return np.trace(Pa) + np.trace(Pb) - 2 * np.real(np.trace(sqrtm(ra @ Pb @ ra)))


checks = {}
def record(name, value):
    checks[name] = float(value)


za = ig.phase_vectors(random_phases(n_trials))
zb = ig.phase_vectors(random_phases(n_trials))
Aa, Ab = ig.factor(za), ig.factor(zb)
Pa, Pb = ig.cos_ipl(za), ig.cos_ipl(zb)
Ua, la, Wa = ig.ipl_eig(za)
Ub, lb, Wb = ig.ipl_eig(zb)

# (1) factorisation and symmetries: P = A A^T; global phase = rotation, conjugation = reflection
record('P = A A^T', np.abs(Pa - Aa @ np.swapaxes(Aa, 1, 2)).max())
phi = rng.uniform(0, 2 * np.pi, n_trials)
record('P invariant to global phase', np.abs(ig.cos_ipl(za * np.exp(1j * phi)[:, None]) - Pa).max())
record('P invariant to conjugation', np.abs(ig.cos_ipl(np.conj(za)) - Pa).max())

# (2) closed-form spectrum, and eigenvectors vs numerical eigendecomposition
w_num = np.linalg.eigvalsh(Pa)[:, ::-1][:, :2]
record('eigenvalues (p +- |sum z^2|)/2', np.abs(w_num - la).max())
record('eigenvalue sum = p', np.abs(la.sum(1) - p).max())
V_num = np.linalg.eigh(Pa)[1][:, :, ::-1][:, :, :2]
record('eigenvectors closed form (1-|cos|)', (1 - np.abs(np.einsum('npq,npq->nq', V_num, Ua))).max())
record('U Lambda^1/2 U^T = P', np.abs(Wa @ np.swapaxes(Wa, 1, 2) - Pa).max())

# (3) eigenvalues are fixed by the eigenvectors (unit diagonal), and P by the span alone
lam_hat = np.array([np.linalg.lstsq(U ** 2, np.ones(p), rcond=None)[0] for U in Ua])
record('eigenvalues recovered from eigenvectors', np.abs(lam_hat - la).max())
err = 0.0
for U, P in zip(Ua[:50], Pa[:50]):
    Q = np.linalg.qr(rng.standard_normal((2, 2)))[0]
    Wr = U @ Q                                   # arbitrary orthonormal basis of the span
    D = np.stack([Wr[:, 0] ** 2, 2 * Wr[:, 0] * Wr[:, 1], Wr[:, 1] ** 2], 1)
    g = np.linalg.lstsq(D, np.ones(p), rcond=None)[0]
    G = np.array([[g[0], g[1]], [g[1], g[2]]])
    err = max(err, np.abs(Wr @ G @ Wr.T - P).max())
record('P recovered from span(U) alone', err)

# (4) sine iPL = det(V) sqrt(l1 l2) (u2 u1^T - u1 u2^T): determined by P up to a global sign
Sa = ig.sin_ipl(za)
S_from_eig = np.sqrt(la[:, 0] * la[:, 1])[:, None, None] * (
    np.einsum('np,nq->npq', Ua[..., 1], Ua[..., 0]) - np.einsum('np,nq->npq', Ua[..., 0], Ua[..., 1]))
record('S = +-sqrt(l1 l2)(u2u1^T - u1u2^T)',
       np.minimum(np.abs(Sa - S_from_eig).max((1, 2)), np.abs(Sa + S_from_eig).max((1, 2))).max())

# (5) BW distance: matrix definition = O(2)-Procrustes on A = weighted Procrustes on U Lambda^1/2
#     = closed form 2p - 2 max(|za^H zb|, |za^T zb|)
k, kc = np.abs(np.sum(np.conj(za) * zb, 1)), np.abs(np.sum(za * zb, 1))
closed = 2 * p - 2 * np.maximum(k, kc)
proc_A = np.array([ig.procrustes_sqdist(Aa[[i]], Ab[[i]])[0, 0] for i in range(n_trials)])
proc_W = np.array([ig.procrustes_sqdist(Wa[[i]], Wb[[i]])[0, 0] for i in range(n_trials)])
bw_mat = np.array([bw_matrix(Pa[i], Pb[i]) for i in range(40)])
record('BW (matrix formula) = closed form', np.abs(bw_mat - closed[:40]).max())
record('Procrustes on A = closed form', np.abs(proc_A - closed).max())
record('weighted Procrustes on eigenvectors = closed form', np.abs(proc_W - closed).max())
record('iplgeom.bw_sqdist = closed form', np.abs(np.diag(ig.bw_sqdist(za, zb)) - closed).max())

# (5b) the same distance in eigenvalue-weighted eigenvector terms (no complex numbers):
#      d^2 = 2p - 2 (s1 + s2), s = singular values of the 2x2 matrix W_a^T W_b,
#      (s1 + s2)^2 = ||W_a^T W_b||_F^2 + 2|det W_a^T W_b| = <P_a,P_b> + |<S_a,S_b>|
M = np.einsum('npq,npr->nqr', Wa, Wb)
sv = np.linalg.svd(M, compute_uv=False)
record('BW = 2p - 2(s1+s2) of W_a^T W_b', np.abs(2 * p - 2 * sv.sum(1) - closed).max())
record('(s1+s2)^2 = ||M||_F^2 + 2|det M|', np.abs(sv.sum(1) ** 2 - (np.sum(M ** 2, (1, 2)) + 2 * np.abs(np.linalg.det(M)))).max())
PP = np.sum(Pa * Pb, (1, 2)); SS = np.sum(ig.sin_ipl(za) * ig.sin_ipl(zb), (1, 2))
record('BW = 2p - 2 sqrt(<Pa,Pb> + |<Sa,Sb>|)', np.abs(2 * p - 2 * np.sqrt(PP + np.abs(SS)) - closed).max())
record('rows of W = unit-length channel phasors', np.abs(np.linalg.norm(Wa, axis=2) - 1).max())

# (5c) LEiDA's leading eigenvector is not injective: mirroring the relative phase of two channels that sit
#      symmetrically around the axial mean phase changes P but leaves u1 (and lambda1) unchanged
phi = rng.uniform(-1.2, 1.2, p); phi[1] = -phi[0]
phi = phi - 0.5 * np.angle(np.sum(np.exp(2j * phi)))      # make 0 the axial mean phase
phi[1] = -phi[0]
phi2 = phi.copy(); phi2[[0, 1]] = phi2[[1, 0]]           # channel 0 and 1 swap lead/lag
z1, z2 = np.exp(1j * phi)[None], np.exp(1j * phi2)[None]
U1, l1, _ = ig.ipl_eig(z1); U2, l2, _ = ig.ipl_eig(z2)
record('LEiDA: same u1 for different P (1-|u1.u1|)', 1 - abs(U1[0, :, 0] @ U2[0, :, 0]))
record('LEiDA: ... while ||P1 - P2||_F is', np.linalg.norm(ig.cos_ipl(z1)[0] - ig.cos_ipl(z2)[0]))

# (6) all existing metrics are functions of (kappa, kappa_conj, r_a, r_b)
ra, rb = ig.axial_order(za), ig.axial_order(zb)
frob = np.sum((Pa - Pb) ** 2, (1, 2))
record('Frobenius = p^2 + (ra^2+rb^2)/2 - k^2 - kc^2', np.abs(frob - (p ** 2 + (ra ** 2 + rb ** 2) / 2 - k ** 2 - kc ** 2)).max())
Ta, Tb = ig.cos_ipl_triu(za), ig.cos_ipl_triu(zb)
cos_eida = np.sum(Ta * Tb, 1) / np.linalg.norm(Ta, axis=1) / np.linalg.norm(Tb, axis=1)
record('EiDA cosine = (k^2+kc^2-2p)/sqrt(...)',
       np.abs(cos_eida - (k ** 2 + kc ** 2 - 2 * p) / np.sqrt((p ** 2 + ra ** 2 - 2 * p) * (p ** 2 + rb ** 2 - 2 * p))).max())
Sb = ig.sin_ipl(zb)
record('<S_a,S_b> = (k^2 - kc^2)/2', np.abs(np.sum(Sa * Sb, (1, 2)) - (k ** 2 - kc ** 2) / 2).max())
record('||S||^2 = (p^2 - r^2)/2', np.abs(np.sum(Sa ** 2, (1, 2)) - (p ** 2 - ra ** 2) / 2).max())

# (7) near-synchrony: theta = eps*g. BW distance between eps and eps+delta depends on delta only;
#     the Frobenius, Grassmann and leading-eigenvector distances shrink with eps.
g = rng.standard_normal(p)
g = (g - g.mean()) / g.std()
delta = 0.05
sens = []
for eps in [0.05, 0.1, 0.2, 0.4, 0.8]:
    z1, z2 = np.exp(1j * eps * g)[None], np.exp(1j * (eps + delta) * g)[None]
    U1, l1, _ = ig.ipl_eig(z1)
    U2, l2, _ = ig.ipl_eig(z2)
    d_bw = np.sqrt(ig.bw_sqdist(z1, z2)[0, 0])
    d_fro = np.linalg.norm(ig.cos_ipl(z1)[0] - ig.cos_ipl(z2)[0])
    d_gr = np.sqrt(max(2 - np.sum((U1[0].T @ U2[0]) ** 2), 0))
    d_le = np.sqrt(max(1 - (U1[0, :, 0] @ U2[0, :, 0]) ** 2, 0))
    s1, s2 = ig.sin_ipl_triu(z1)[0], ig.sin_ipl_triu(z2)[0]
    d_si = np.sqrt(max(1 - (s1 @ s2 / np.linalg.norm(s1) / np.linalg.norm(s2)) ** 2, 0))
    # each distance normalised by its maximum so that the four are on a common [0, 1] scale
    sens.append(dict(eps=eps, bw=d_bw / np.sqrt(2 * p), frobenius=d_fro / (np.sqrt(2) * p), grassmann=d_gr / np.sqrt(2), leida=d_le, sine=d_si))
record('BW near-sync: d / (delta ||g||) at eps=0.05', sens[0]['bw'] * np.sqrt(2 * p) / (delta * np.linalg.norm(g)))

if __name__ == '__main__':
    width = max(len(n) for n in checks)
    print('Identity'.ljust(width), ' max abs error')
    for name, v in checks.items():
        print(name.ljust(width), f' {v:.2e}')
    print('\nNear-synchrony: distance between theta = eps*g and (eps+0.05)*g (normalised; ratio to BW)')
    print(f'{"eps":>6}{"BW":>9}{"Frobenius":>18}{"Grassmann":>18}{"LEiDA":>18}{"Sine":>18}')
    for s in sens:
        print(f'{s["eps"]:6.2f}{s["bw"]:9.4f}' + ''.join(f'{s[m]:10.4f} ({s[m] / s["bw"]:4.2f})' for m in ['frobenius', 'grassmann', 'leida', 'sine']))
    os.makedirs(RES, exist_ok=True)
    json.dump({'checks': checks, 'near_sync': sens}, open(os.path.join(RES, 'theory_checks.json'), 'w'), indent=1)
