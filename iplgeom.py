"""Geometry and clustering of instantaneous phase-locking (iPL) matrices.

Notation (follows the paper):
    X      (p, T)    real signals, p channels
    theta  (T, p)    instantaneous phases from the analytic signal
    z      (T, p)    unit-modulus phase vectors, z_t = exp(i theta_t)
    A      (T, p, 2) real factor A_t = [cos theta_t, sin theta_t] = [Re z_t, Im z_t]
    P      cosine iPL, P_t = A_t A_t^T = Re(z_t z_t^H)   (p x p, PSD, rank <= 2, unit diagonal)
    S      sine iPL,   S_t = Im(z_t z_t^H)

Every distance used by the compared methods is computed here from z directly. The paper writes the
BW distance in real terms, d^2 = 2P - 2(s1 + s2) with s the singular values of Ut_a^T Ut_b, where
Ut = [sqrt(l1) u1, sqrt(l2) u2] (orthogonal Procrustes on eigenvalue-weighted eigenvectors). Because
the rows of Ut are the channel phasors [Re z, Im z] up to an O(2) transform, this equals bw_sqdist below
(verified to 1e-13; see verify_theory.py and code/matlab for the real-valued implementation).
"""
import numpy as np
import scipy.sparse as sp
from scipy.signal import butter, filtfilt, hilbert


# --------------------------------------------------------------------------- signal processing
def instantaneous_phase(X, fs, band, order=3, decim=1, trim=0):
    """Band-pass (zero-phase Butterworth) -> FFT-based analytic signal -> phase.

    X: (p, T) real signals. The analytic signal is computed at the original rate and then
    decimated by `decim` (the band-limited signal has no content above the new Nyquist rate).
    `trim` samples (after decimation) are removed at both ends to limit Hilbert edge effects.
    Returns theta with shape (T', p).
    """
    b, a = butter(order, np.asarray(band, float) / (fs / 2), btype='band')
    Xf = filtfilt(b, a, np.asarray(X, float), axis=1)
    theta = np.angle(hilbert(Xf, axis=1))[:, ::decim].T
    if trim:
        theta = theta[trim:-trim]
    return theta


# --------------------------------------------------------------------------- representations
def phase_vectors(theta):
    return np.exp(1j * np.asarray(theta))


def factor(z):
    """A = [Re z, Im z]; P = A A^T."""
    return np.stack([z.real, z.imag], axis=-1)


def cos_ipl(z):
    return np.real(z[..., :, None] * np.conj(z[..., None, :]))


def sin_ipl(z):
    return np.imag(z[..., :, None] * np.conj(z[..., None, :]))


def triu_pairs(p):
    return np.triu_indices(p, 1)


def cos_ipl_triu(z):
    """Upper triangle (j<k) of the cosine iPL, cos(theta_j - theta_k), without forming P."""
    j, k = triu_pairs(z.shape[-1])
    return np.real(z[:, j] * np.conj(z[:, k]))


def sin_ipl_triu(z):
    j, k = triu_pairs(z.shape[-1])
    return np.imag(z[:, j] * np.conj(z[:, k]))


def ipl_eig(z):
    """Closed-form eigendecomposition of P = Re(z z^H).

    With m = sum_j z_j^2 and psi = arg m:
        lambda_{1,2} = (p +- |m|) / 2,
        u_1 ∝ cos(theta - psi/2),  u_2 ∝ sin(theta - psi/2),
    i.e. U Lambda^{1/2} = [Re zt, Im zt] with zt = exp(-i psi/2) z.
    Returns U (n,p,2) orthonormal eigenvectors, lam (n,2), W = U Lambda^{1/2} (n,p,2).
    """
    z = np.atleast_2d(z)
    p = z.shape[-1]
    m = np.sum(z ** 2, axis=-1)
    W = factor(z * np.exp(-0.5j * np.angle(m))[:, None])
    lam = np.stack([(p + np.abs(m)) / 2, (p - np.abs(m)) / 2], axis=-1)
    U = W / np.sqrt(np.maximum(lam, 1e-300))[:, None, :]
    return U, lam, W


def kappa(Za, Zb):
    """kappa = |z_a^H z_b| (match up to a global phase) and
    kappa_conj = |z_a^T z_b| (match after conjugating one pattern), for all row pairs."""
    return np.abs(np.conj(Za) @ Zb.T), np.abs(Za @ Zb.T)


def axial_order(z):
    """r = |sum_j z_j^2| = lambda_1 - lambda_2."""
    return np.abs(np.sum(z ** 2, axis=-1))


# --------------------------------------------------------------------------- distances
def bw_sqdist(Za, Zb):
    """Squared Bures-Wasserstein distance between Re(z_a z_a^H) and Re(z_b z_b^H):
        ||z_a||^2 + ||z_b||^2 - 2 max(|z_a^H z_b|, |z_a^T z_b|).
    Valid for any complex vectors (e.g. cluster centroids), not only unit-modulus ones."""
    k, kc = kappa(Za, Zb)
    na = np.sum(np.abs(Za) ** 2, axis=-1)
    nb = np.sum(np.abs(Zb) ** 2, axis=-1)
    return np.maximum(na[:, None] + nb[None, :] - 2 * np.maximum(k, kc), 0.0)


def procrustes_sqdist(Fa, Fb):
    """min_{R in O(2)} ||F_a - F_b R||_F^2 for factors F (n,p,2), via the 2x2 nuclear norm."""
    a = Fa[..., 0] @ Fb[..., 0].T
    b = Fa[..., 0] @ Fb[..., 1].T
    c = Fa[..., 1] @ Fb[..., 0].T
    d = Fa[..., 1] @ Fb[..., 1].T
    nuc = np.maximum(np.hypot(a + d, c - b), np.hypot(a - d, b + c))
    na = np.sum(Fa ** 2, axis=(1, 2))
    nb = np.sum(Fb ** 2, axis=(1, 2))
    return np.maximum(na[:, None] + nb[None, :] - 2 * nuc, 0.0)


# --------------------------------------------------------------------------- clustering methods
def _normalize(X):
    tiny = np.finfo(np.result_type(X.dtype, np.float32)).tiny     # 1e-300 underflows in float32
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), tiny)


def _top_eig(M, q):
    w, V = np.linalg.eigh((M + M.T) / 2)
    return w[::-1][:q], V[:, ::-1][:, :q]


def _frame_overlap_sq(X, C):
    """||X_n^T C_k||_F^2 for frames X (n,p,2) and C (K,p,2) -> (n,K)."""
    out = 0.0
    for q in range(2):
        for r in range(2):
            out = out + (X[..., q] @ C[..., r].T) ** 2
    return out


class Method:
    """A K-means-type method: representation, distance to centroids, centroid update.
    `batch_update(X, labels, C)` optionally updates all centroids in one pass over X
    (used for the high-dimensional vectorized methods; equivalent to the per-cluster update)."""

    def __init__(self, name, label, prepare, dist, update, batch_update=None):
        self.name, self.label = name, label
        self.prepare, self.dist, self.update = prepare, dist, update
        self.batch_update = batch_update


def _onehot(labels, K):
    n = len(labels)
    return sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, K))


def _masked(Y, labels):
    W = np.zeros_like(Y)
    idx = np.arange(len(labels))
    W[idx, labels] = Y[idx, labels]
    return W


def _batch_spherical_mean(X, labels, C):
    M = np.asarray(_onehot(labels, len(C)).T @ X, dtype=X.dtype)
    return _normalize(M)


def _batch_diametrical(X, labels, C):
    # Dhillon et al. (2003) power step for every cluster at once: c_k <- sum_{t in k} x_t (x_t^T c_k)
    V = (X.T @ _masked(X @ C.T, labels)).T
    return _normalize(V)


def _diametrical_update(Xk, c):
    # Dhillon et al. (2003): one power step on the within-cluster scatter
    v = Xk.T @ (Xk @ c)
    return v / max(np.linalg.norm(v), 1e-300)


def _bw_update(Zk, c, n_inner=3):
    """Barycentre step for BW K-means (generalized Procrustes analysis, Gower 1975):
    align every member to the centroid with the optimal O(2) element (global phase and,
    if better, conjugation) and average the aligned phase vectors."""
    for _ in range(n_inner):
        s = np.conj(Zk) @ c          # conj(c^H z)
        sc = Zk @ c                  # c^T z
        flip = np.abs(sc) > np.abs(s)
        Zt = np.where(flip[:, None], np.conj(Zk), Zk)
        h = Zt @ np.conj(c)          # c^H zt
        c = np.mean(Zt * np.exp(-1j * np.angle(h))[:, None], axis=0)
    return c


def _frob_dist(X, C):
    nx = np.sum((np.einsum('npq,npr->nqr', X, X)) ** 2, axis=(1, 2))
    nc = np.sum((np.einsum('kpq,kpr->kqr', C, C)) ** 2, axis=(1, 2))
    return np.maximum(nx[:, None] + nc[None, :] - 2 * _frame_overlap_sq(X, C), 0.0)


def _frame_scatter(Xk):
    """sum_n X_n X_n^T for frames X (n,p,2); contiguous reshape keeps the product in BLAS."""
    F = np.ascontiguousarray(np.moveaxis(Xk, 1, 0)).reshape(Xk.shape[1], -1)
    return F @ F.T


def _frob_update(Xk, c):
    M = _frame_scatter(Xk) / len(Xk)
    w, V = _top_eig(M, 2)
    return V * np.sqrt(np.maximum(w, 0))[None, :]


def _grass_update(Xk, c):
    M = _frame_scatter(Xk)
    return _top_eig(M, 2)[1]


METHODS = {
    # leading eigenvector only, sign-invariant (diametrical) clustering
    'leida': Method('leida', 'LEiDA',
                    lambda z: ipl_eig(z)[0][..., 0],
                    lambda X, C: 1 - (X @ C.T) ** 2,
                    _diametrical_update),
    # discrete EiDA: k-means with cosine distance on the vectorized iPL
    'eida': Method('eida', 'EiDA',
                   lambda z: _normalize(cos_ipl_triu(z)).astype(np.float32),
                   lambda X, C: 1 - X @ C.T,
                   lambda Xk, c: _normalize(Xk.mean(0)), _batch_spherical_mean),
    # Frobenius distance on P (DySCo distance-2; weighted Grassmann clustering)
    'frobenius': Method('frobenius', 'Frobenius',
                        lambda z: factor(z),
                        _frob_dist,
                        _frob_update),
    # Grassmann (projection) distance between the spans of the eigenvectors
    'grassmann': Method('grassmann', 'Grassmann',
                        lambda z: ipl_eig(z)[0],
                        lambda X, C: np.maximum(2 - _frame_overlap_sq(X, C), 0.0),
                        _grass_update),
    # sine iPL (imaginary part), sign-invariant cosine similarity
    'sine': Method('sine', 'Sine iPL',
                   lambda z: _normalize(sin_ipl_triu(z)).astype(np.float32),
                   lambda X, C: 1 - (X @ C.T) ** 2,
                   _diametrical_update, _batch_diametrical),
    # proposed: Bures-Wasserstein K-means
    'bw': Method('bw', 'BW (proposed)',
                 lambda z: np.asarray(z, complex),
                 bw_sqdist,
                 _bw_update),
    # reference that uses the complex phase vector itself (not a function of P)
    'complex': Method('complex', 'Complex diam. (ref.)',
                      lambda z: np.asarray(z, complex) / np.sqrt(z.shape[-1]),
                      lambda X, C: 1 - np.abs(X @ np.conj(C).T) ** 2,
                      lambda Xk, c: _normalize(Xk.T @ (np.conj(Xk) @ c))),
}
COSINE_METHODS = ['leida', 'eida', 'frobenius', 'grassmann', 'sine', 'bw']


def _plusplus(method, X, K, rng):
    n = len(X)
    idx = [int(rng.integers(n))]
    dmin = method.dist(X, X[idx])[:, 0]
    for _ in range(K - 1):
        p = np.maximum(dmin, 0)
        p = p / p.sum() if p.sum() > 0 else np.full(n, 1 / n)
        i = int(rng.choice(n, p=p))
        idx.append(i)
        dmin = np.minimum(dmin, method.dist(X, X[[i]])[:, 0])
    return X[idx].copy()


def fit(method, X, K, n_init=10, max_iter=300, tol=1e-10, seed=0):
    """K-means in the geometry of `method` (Lloyd iterations, ++ initialization, n_init restarts).
    X must already be in the method's representation (method.prepare(z)).
    Returns (labels, centroids, mean distance to assigned centroid)."""
    if isinstance(method, str):
        method = METHODS[method]
    rng = np.random.default_rng(seed)
    n = len(X)
    best = None
    for _ in range(n_init):
        C = _plusplus(method, X, K, rng)
        prev_lab, prev_obj = None, np.inf
        for _ in range(max_iter):
            D = method.dist(X, C)
            lab = D.argmin(1)
            obj = D[np.arange(n), lab].mean()
            if prev_lab is not None and np.array_equal(lab, prev_lab) and prev_obj - obj <= tol * abs(prev_obj):
                break
            counts = np.bincount(lab, minlength=K)
            if method.batch_update is not None:
                C = method.batch_update(X, lab, C)
            for k in range(K):
                if counts[k] == 0:              # re-seed an empty cluster with the worst-fit sample
                    C[k] = X[D[np.arange(n), lab].argmax()]
                elif method.batch_update is None:
                    C[k] = method.update(X[lab == k], C[k])
            prev_lab, prev_obj = lab, obj
        D = method.dist(X, C)
        lab = D.argmin(1)
        obj = D[np.arange(n), lab].mean()
        if best is None or obj < best[2]:
            best = (lab, C.copy(), obj)
    return best


def assign(method, z, C, block=20000):
    """Assign phase vectors z (n,p) to centroids C of `method`, in blocks."""
    if isinstance(method, str):
        method = METHODS[method]
    labels = np.empty(len(z), int)
    for s in range(0, len(z), block):
        labels[s:s + block] = method.dist(method.prepare(z[s:s + block]), C).argmin(1)
    return labels


def bw_barycentre(z, n_iter=100, tol=1e-12):
    """BW barycentre (Fréchet mean) of the iPL matrices Re(z z^H) of a set of samples,
    returned as a complex vector c with P_c = Re(c c^H): |c_j| is the phase consistency of
    channel j and arg c_j its relative phase (both defined up to global phase and conjugation)."""
    z = np.asarray(z, complex)
    w, V = np.linalg.eigh(z.T @ np.conj(z))
    c = V[:, -1] * np.sqrt(max(w[-1], 0) / len(z))
    prev = np.inf
    for _ in range(n_iter):
        c = _bw_update(z, c, n_inner=1)
        obj = bw_sqdist(z, c[None])[:, 0].mean()
        if prev - obj <= tol * abs(prev):
            break
        prev = obj
    return c


def canonical_phase_map(c, xy=None):
    """Relative phase map (radians) of a complex pattern c, with the global phase set so that the
    consistency-weighted mean phase is zero and the conjugation ambiguity fixed by requiring a
    positive slope along the anterior-posterior axis (xy[:,1]) when a layout is given."""
    c = c * np.exp(-1j * np.angle(np.sum(c)))
    phi = np.angle(c)
    if xy is not None:
        slope = np.polyfit(xy[:, 1], phi, 1)[0]
        if slope < 0:
            phi = -phi
    return phi


# --------------------------------------------------------------------------- simulation
def simulate(templates, n_per, sigma, rng):
    """Phase vectors from K template phase patterns (K,p): theta = template + global phase + noise.
    The global phase is uniform on [0, 2pi) (it carries no information about the state) and the
    noise is i.i.d. Gaussian with s.d. `sigma` (radians) on every channel."""
    K, p = templates.shape
    theta = np.repeat(templates, n_per, axis=0)
    theta = theta + rng.uniform(0, 2 * np.pi, (K * n_per, 1)) + sigma * rng.standard_normal((K * n_per, p))
    return np.exp(1j * theta), np.repeat(np.arange(K), n_per)


# --------------------------------------------------------------------------- state statistics
def dwell_times(labels_2d, K, dt):
    """Mean dwell time (s) per state from labels arranged as (subjects, time)."""
    out = np.zeros(K)
    for k in range(K):
        runs = []
        for row in labels_2d:
            x = np.concatenate([[0], (row == k).astype(int), [0]])
            d = np.diff(x)
            runs.extend(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
        out[k] = np.mean(runs) * dt if runs else 0.0
    return out
