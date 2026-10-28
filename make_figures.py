"""Figures for the paper (Fig. 1 synthetic, Fig. 2 EEG) and supplementary panels for the notebook.

Run:  python make_figures.py   (after synthetic_experiment.py and eeg_experiment.py)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import RBFInterpolator

sys.path.insert(0, os.path.dirname(__file__))
import iplgeom as ig
from eeg_layout import XY

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
EEG_TAG = os.environ.get('EEG_TAG', '')          # '_smoke' to test on the smoke-test output
FIG = os.path.join(HERE, '..', 'figures')

# --- palette (validated reference palette, categorical slots in fixed order; ink and grid roles)
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
STYLE = {   # method -> (label, colour, marker); BW takes slot 1, the complex reference is neutral ink
    'bw':        ('BW (proposed)', '#2a78d6', 'o'),
    'leida':     ('LEiDA', '#eb6834', 's'),
    'eida':      ('EiDA', '#1baf7a', '^'),
    'frobenius': ('Frobenius', '#eda100', 'D'),
    'grassmann': ('Grassmann', '#e87ba4', 'v'),
    'sine':      ('Sine iPL', '#008300', 'P'),
    'complex':   ('Complex (ref.)', MUTED, None),
    'bw_medoids': ('BW medoids', '#6250d6', 'x'),
}
ORDER = ['bw', 'leida', 'eida', 'frobenius', 'grassmann', 'sine', 'complex']
# relative phase is circular: EEG maps use a cyclic map whose +-180 deg ends meet (rotating waves stay continuous)
PHASE_CYCLIC = plt.get_cmap('twilight_shifted')


def _lab_circle(L=60, C=58, h0=25, n=256):
    """Cyclic map of constant CIELAB lightness and chroma (no white or black anywhere), hue h0 at 0 rad,
    so that zero phase is a saturated colour and stays distinct from the fade toward low consistency."""
    h = np.deg2rad(h0 + np.linspace(0, 360, n, endpoint=False))
    a, b = C * np.cos(h), C * np.sin(h)
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, np.full(n, fy), fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])[:, None]
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]) @ xyz
    rgb = np.where(rgb > 0.0031308, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055, 12.92 * rgb)
    rgb = np.clip(rgb.T, 0, 1)
    return LinearSegmentedColormap.from_list('phase_lab', np.roll(rgb, n // 2, axis=0), N=n)


PHASE_SAT = _lab_circle()
# Fig. 3: red = in phase, blue = anti-phase; the two quadrature phases differ in lightness as well as hue
# (+pi/2 green, -pi/2 near-black), so leads and lags are told apart at a glance
PHASE_RB = LinearSegmentedColormap.from_list('phase_rb', [
    (0.000, '#1f5fbf'), (0.125, '#4b2c91'), (0.250, '#1e0f3c'), (0.375, '#b8297e'), (0.500, '#d62f27'),
    (0.625, '#f08a1c'), (0.750, '#2e9e3e'), (0.875, '#16a3b5'), (1.000, '#1f5fbf')], N=512)
CYCLIC_MAPS = (PHASE_CYCLIC, PHASE_SAT, PHASE_RB)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    'phase', ['#104281', '#2a78d6', '#9ec5f4', '#f0efec', '#f2b0af', '#e34948', '#8f1f1f'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

_GX, _GY = np.meshgrid(np.linspace(-1.05, 1.05, 121), np.linspace(-1.05, 1.05, 121))
_INSIDE = _GX ** 2 + _GY ** 2 <= 1.02 ** 2


def _interp(values):
    """Thin-plate-spline interpolation of channel values onto the head grid (smooth up to the rim)."""
    rbf = RBFInterpolator(XY, values, kernel='thin_plate_spline', smoothing=1e-3)
    return rbf(np.column_stack([_GX.ravel(), _GY.ravel()])).reshape(_GX.shape)


def topomap(ax, values, vmax, cmap=PHASE_CMAP, title=None, dots=True, weight=None):
    """Scalp map: cubic interpolation inside the electrode hull, nearest-neighbour outside, head mask.
    `weight` in [0, 1] per channel fades the colour toward the surface (used for phase consistency,
    so that the arbitrary phase of incoherent channels is not shown at full strength)."""
    if cmap in CYCLIC_MAPS:           # interpolate phase (degrees) on the circle, not across the wrap
        rad = np.deg2rad(values)
        f = np.rad2deg(np.arctan2(_interp(np.sin(rad)), _interp(np.cos(rad))))
    else:
        f = _interp(values)
    rgba = cmap(plt.Normalize(-vmax, vmax)(f))
    if weight is not None:
        w = np.clip(_interp(weight), 0, 1)[..., None]
        rgba[..., :3] = w * rgba[..., :3] + (1 - w) * np.array(matplotlib.colors.to_rgb('#fcfcfb'))
    rgba[~_INSIDE] = (1, 1, 1, 0)
    ax.imshow(rgba, extent=(-1.05, 1.05, -1.05, 1.05), origin='lower', interpolation='bilinear')
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(1.03 * np.cos(t), 1.03 * np.sin(t), color=INK2, lw=0.5)
    ax.plot([-0.12, 0, 0.12], [1.02, 1.16, 1.02], color=INK2, lw=0.5)
    if dots:
        ax.scatter(XY[:, 0], XY[:, 1], s=0.4, c=INK2, linewidths=0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.1, 1.22)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=6, color=INK2, pad=1)


def _lines(ax, x, data, methods, key, ylabel):
    for m in methods:
        label, col, mk = STYLE[m]
        if f'{key}{m}_nmi' not in data:
            continue
        y = data[f'{key}{m}_nmi']
        if np.all(np.isnan(y)):
            continue
        mu, sd = np.nanmean(y, 1), np.nanstd(y, 1)
        if m == 'complex':
            ax.plot(x, mu, ls='--', color=col, lw=1.0, label=label, zorder=2)
            continue
        lw = 2.0 if m == 'bw' else 1.2
        ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.10, lw=0)
        ax.plot(x, mu, color=col, lw=lw, marker=mk, ms=3.2, mew=0.6, mec='#fcfcfb', label=label,
                zorder=4 if m == 'bw' else 3)
    ax.set_ylabel(ylabel)


ORDER_PAPER = ['bw', 'leida', 'eida', 'frobenius', 'grassmann', 'sine']   # cosine-iPL methods only


def fig_synthetic(path=os.path.join(FIG, 'fig2_synthetic')):
    d = np.load(os.path.join(RES, 'synthetic.npz'))
    sig = d['sigmas']
    templates = d['templates_main']
    vmax = float(d['large']) / 2                                   # radians
    fig = plt.figure(figsize=(3.45, 2.85))
    top = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.07], left=0.10, right=0.92, top=0.95, bottom=0.71,
                           wspace=0.08)
    names = ['AP, small lag', 'AP, large lag', 'LR, small lag', 'LR, large lag']
    for k in range(4):
        topomap(fig.add_subplot(top[0, k]), templates[k] - templates[k].mean(), vmax, title=names[k])
    cax = fig.add_subplot(top[0, 4])
    sm = plt.cm.ScalarMappable(cmap=PHASE_CMAP, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels([f'{-vmax:.2f}', '0', f'{vmax:.2f}'])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.set_label('phase (rad)', fontsize=5.5, labelpad=1)
    ax = fig.add_axes([0.13, 0.25, 0.79, 0.40])
    _lines(ax, sig, d, ORDER_PAPER, 'main_', 'NMI with true states')
    ax.set_xlabel(r'phase-noise s.d. $\sigma$ (rad)', labelpad=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(sig[0] - 0.02, sig[-1] + 0.02)
    ax.legend(ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.27), handlelength=1.8,
              columnspacing=0.9, borderaxespad=0.0)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg(path=os.path.join(FIG, 'fig2_eeg'), K=4, rows=('leida', 'eida', 'grassmann', 'bw')):
    d = np.load(os.path.join(RES, f'eeg{EEG_TAG}.npz'))
    s = json.load(open(os.path.join(RES, f'eeg_summary{EEG_TAG}.json')))
    ks = d['ks']
    fig = plt.figure(figsize=(7.1, 2.55))
    left = fig.add_gridspec(len(rows), K, left=0.03, right=0.465, bottom=0.02, top=0.88, hspace=0.32, wspace=0.02)
    cent = {m: d[f'{m}_K{K}_maps'] for m in rows}
    maps = {m: np.stack([np.rad2deg(ig.canonical_phase_map(c, XY)) for c in cent[m]]) for m in rows}
    vmax = 180
    for i, m in enumerate(rows):
        occ = s['methods'][m][str(K)]['occupancy']
        for k in range(K):
            ax = fig.add_subplot(left[i, k])
            cons = np.abs(cent[m][k])
            topomap(ax, maps[m][k], vmax, cmap=PHASE_CYCLIC, title=f'{100 * occ[k]:.0f}%',
                    weight=2 * cons / cons.max())
            if k == 0:
                ax.text(-1.45, 0, STYLE[m][0].replace(' (proposed)', ''), rotation=90, ha='center',
                        va='center', fontsize=6.5, color=INK, fontweight='bold' if m == 'bw' else 'normal')
    cax = fig.add_axes([0.478, 0.12, 0.007, 0.68])
    sm = plt.cm.ScalarMappable(cmap=PHASE_CYCLIC, norm=plt.Normalize(-vmax, vmax))
    cb = fig.colorbar(sm, cax=cax, ticks=[-180, -90, 0, 90, 180])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.ax.set_title('phase\n(°)', fontsize=5.5, color=INK2, pad=3)
    for j, (key, ylabel, title) in enumerate([
            ('occupancy', 'occupancy', '(b) largest state'),
            ('split_half_ari_mean', 'ARI', '(c) split-half reproducibility')]):
        ax = fig.add_axes([0.585 + j * 0.175, 0.17, 0.125, 0.68])
        ax.set_ylabel(ylabel, labelpad=1)
        for m in ORDER:
            label, col, mk = STYLE[m]
            y = [s['methods'][m][str(K_)][key] for K_ in ks]
            y = [v[0] if isinstance(v, list) else v for v in y]
            if m == 'complex':
                ax.plot(ks, y, ls='--', color=col, lw=1.0, label=label)
            else:
                ax.plot(ks, y, color=col, lw=2.0 if m == 'bw' else 1.2, marker=mk, ms=3.2, mew=0.6,
                        mec='#fcfcfb', label=label, zorder=4 if m == 'bw' else 3)
        if key == 'occupancy':
            ax.plot(ks, 1 / ks, color=AXIS, lw=0.8, ls=':', label='uniform, 1/K')
            handles, labels = ax.get_legend_handles_labels()
        ax.set_xlabel('number of states K')
        ax.set_title(title, loc='left', color=INK)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1)
    lax = fig.add_axes([0.89, 0.17, 0.1, 0.68])
    lax.axis('off')
    lax.legend(handles, labels, frameon=False, loc='center left', fontsize=6, handlelength=1.8)
    fig.text(0.03, 0.965, f'(a) phase-locking states, K = {K} (BW barycentre of each state)',
             fontsize=7, color=INK, ha='left')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_methods(path=os.path.join(FIG, 'fig1_methods')):
    """Pipeline figure: spatial phase pattern -> travelling alpha wave -> channel phasors = rows of
    W = U Lambda^1/2 -> Procrustes/BW alignment."""
    from eeg_layout import LABELS
    ramp = ['#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']        # Oz -> Fz
    names = ['Oz', 'Poz', 'Pz', 'Cz', 'Fz']
    idx = [LABELS.index(n) for n in names]
    g = -XY[:, 1] / 0.8                                                   # AP phase pattern, posterior leads
    amp = {'small': 0.25, 'large': 0.75}                                  # radians at Oz
    f, t = 10.0, np.linspace(0, 0.2, 400)
    t0 = 0.12
    fig = plt.figure(figsize=(3.45, 2.75))
    top = fig.add_gridspec(1, 2, width_ratios=[0.34, 0.66], left=0.02, right=0.98, top=0.92, bottom=0.60, wspace=0.18)
    bot = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.47, bottom=0.04, wspace=0.42)
    # (a) relative-phase topography with the five midline electrodes
    ax = fig.add_subplot(top[0, 0])
    topomap(ax, amp['large'] * g, amp['large'], dots=False)
    for j, i in enumerate(idx):
        ax.plot(XY[i, 0], XY[i, 1], 'o', ms=3.6, color=ramp[j], mec=INK, mew=0.5, zorder=5)
    ax.set_title('(a) phase across scalp', loc='left', fontsize=6.5, color=INK, pad=2)
    # (b) waveforms
    ax = fig.add_subplot(top[0, 1])
    for j, i in enumerate(idx):
        th = amp['large'] * g[i]
        ax.plot(t * 1000, np.cos(2 * np.pi * f * t + th) + 2.4 * j, color=ramp[j], lw=1.1)
        ax.text(-5, 2.4 * j, names[j].replace('Poz', 'POz'), ha='right', va='center', fontsize=6, color=INK2)
        ax.plot(t0 * 1000, np.cos(2 * np.pi * f * t0 + th) + 2.4 * j, 'o', ms=3.2, color=ramp[j],
                mec='#fcfcfb', mew=0.6)
    ax.axvline(t0 * 1000, color=MUTED, lw=0.6, ls='--')
    ax.set_yticks([]); ax.set_xlim(0, 200); ax.set_xlabel('time (ms)', labelpad=1)
    ax.spines['left'].set_visible(False); ax.grid(False)
    ax.set_title('(b) 10 Hz wave travelling Oz$\\to$Fz', loc='left', fontsize=6.5, color=INK, pad=2)

    def phasor_panel(ax, W, title, lam):
        tt = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(tt), np.sin(tt), color=GRID, lw=0.8, zorder=1)
        ax.axhline(0, color=AXIS, lw=0.5, zorder=1); ax.axvline(0, color=AXIS, lw=0.5, zorder=1)
        for j in range(len(W)):
            ax.plot([0, W[j, 0]], [0, W[j, 1]], color=ramp[j], lw=0.8, zorder=2)
            ax.plot(W[j, 0], W[j, 1], 'o', ms=3.6, color=ramp[j], mec='#fcfcfb', mew=0.6, zorder=3)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(r'$\sqrt{\lambda_1}\,\mathbf{u}_1$', labelpad=0, fontsize=6)
        ax.set_ylabel(r'$\sqrt{\lambda_2}\,\mathbf{u}_2$', labelpad=0, fontsize=6)
        ax.set_title(title, loc='left', fontsize=6.5, color=INK)
        ax.text(-1.12, 1.12, rf'$\lambda_1/P={lam:.2f}$', ha='left', va='top', fontsize=5.5, color=INK2)

    Ws = {}
    for col, (k, label) in enumerate([('small', '(c) small lag'), ('large', '(d) large lag')]):
        z = np.exp(1j * amp[k] * g[idx])[None]
        U, lam, W = ig.ipl_eig(z)
        W = W[0] * np.sign(W[0][:, 0].sum())                              # resolve the eigenvector sign
        Ws[k] = W
        phasor_panel(fig.add_subplot(bot[0, col]), W, label, lam[0, 0] / len(idx))
    # (e) Procrustes alignment of the large-lag pair onto the small-lag pair
    Wa, Wb = Ws['small'], Ws['large']
    Q, _, Rt = np.linalg.svd(Wb.T @ Wa)
    Wb_al = Wb @ (Q @ Rt)
    ax = fig.add_subplot(bot[0, 2])
    tt = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(tt), np.sin(tt), color=GRID, lw=0.8, zorder=1)
    ax.axhline(0, color=AXIS, lw=0.5, zorder=1); ax.axvline(0, color=AXIS, lw=0.5, zorder=1)
    for j in range(len(idx)):
        ax.plot([Wa[j, 0], Wb_al[j, 0]], [Wa[j, 1], Wb_al[j, 1]], color=INK, lw=1.2, zorder=2,
                solid_capstyle='round')
        ax.plot(Wa[j, 0], Wa[j, 1], 'o', ms=3.6, color=ramp[j], mec='#fcfcfb', mew=0.6, zorder=3)
        ax.plot(Wb_al[j, 0], Wb_al[j, 1], 'o', ms=3.6, mfc='#fcfcfb', mec=ramp[j], mew=1.0, zorder=3)
    d2 = np.sum((Wa - Wb_al) ** 2)
    ax.text(-1.12, 1.12, 'filled: (c)\nopen: (d), rotated', ha='left', va='top', fontsize=5.3, color=INK2)
    ax.text(0.0, -1.12, rf'$d^2_{{\mathrm{{BW}}}}={d2:.2f}$', ha='center', va='bottom', fontsize=6, color=INK2)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('(e) Procrustes / BW', loc='left', fontsize=6.5, color=INK)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg_states(path=os.path.join(FIG, 'old_v1', 'fig3_eeg_pooled'), K=4, rows=('leida', 'grassmann', 'bw')):
    """One-column EEG figure: K = 4 states of LEiDA, Grassmann and BW K-means (eyes open + closed pooled)."""
    d = np.load(os.path.join(RES, 'eoec.npz'))
    s = json.load(open(os.path.join(RES, 'eoec_summary.json')))
    fig = plt.figure(figsize=(3.45, 2.5))
    gs = fig.add_gridspec(len(rows), K, left=0.09, right=0.86, top=0.93, bottom=0.02, hspace=0.30, wspace=0.04)
    labels = {'leida': 'LEiDA', 'grassmann': 'Grassmann', 'bw': 'BW'}
    for i, m in enumerate(rows):
        occ = s['methods'][m][str(K)]['occupancy']
        for k in range(K):
            c = d[f'{m}_K{K}_maps'][k]
            cons = np.abs(c)
            ax = fig.add_subplot(gs[i, k])
            topomap(ax, np.rad2deg(ig.canonical_phase_map(c, XY)), 180, cmap=PHASE_CYCLIC,
                    title=f'{100 * occ[k]:.0f}%', weight=2 * cons / cons.max())
            if k == 0:
                ax.text(-1.5, 0, labels[m], rotation=90, ha='center', va='center', fontsize=6.5, color=INK,
                        fontweight='bold' if m == 'bw' else 'normal')
    cax = fig.add_axes([0.89, 0.15, 0.022, 0.65])
    sm = plt.cm.ScalarMappable(cmap=PHASE_CYCLIC, norm=plt.Normalize(-np.pi, np.pi))
    cb = fig.colorbar(sm, cax=cax, ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.ax.set_title('phase\n(rad)', fontsize=5.5, color=INK2, pad=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg_states_saturated(path=os.path.join(FIG, 'fig3_eeg_colour'), K=4, rows=('leida', 'grassmann', 'bw'),
                             variant=None, cmap=PHASE_SAT):
    """Fig. 3 with a constant-lightness cyclic map: zero phase is saturated, low consistency fades to white.
    `variant` (e.g. 'ec_native') takes the states from eeg_variants.py and adds mean lambda_1/P to the titles."""
    if variant:
        v = np.load(os.path.join(RES, 'variants.npz'))
        vs = json.load(open(os.path.join(RES, 'variants_summary.json')))
        d = {f'{m}_K{K}_maps': v[f'{variant}_{m}_K{K}_maps'] for m in rows}
        s = {'methods': {m: {str(K): {'occupancy': vs[f'{variant}_{m}_K{K}']['occupancy']}} for m in rows}}
        lam = {m: vs[f'{variant}_{m}_K{K}']['lambda1_over_p'] for m in rows}
    else:
        d = np.load(os.path.join(RES, 'eoec.npz'))
        s = json.load(open(os.path.join(RES, 'eoec_summary.json')))
    fig = plt.figure(figsize=(3.45, 2.5))
    gs = fig.add_gridspec(len(rows), K, left=0.09, right=0.835, top=0.93, bottom=0.02, hspace=0.30, wspace=0.04)
    labels = {'leida': 'LEiDA', 'grassmann': 'Grassmann', 'bw': 'BW'}
    for i, m in enumerate(rows):
        occ = s['methods'][m][str(K)]['occupancy']
        for k in range(K):
            c = d[f'{m}_K{K}_maps'][k]
            ax = fig.add_subplot(gs[i, k])
            title = f'{100 * occ[k]:.0f}%' + (f'  ({lam[m][k]:.2f})' if variant else '')
            topomap(ax, np.rad2deg(ig.canonical_phase_map(c, XY)), 180, cmap=cmap,
                    title=title, weight=2 * np.abs(c) / np.abs(c).max())
            if k == 0:
                ax.text(-1.5, 0, labels[m], rotation=90, ha='center', va='center', fontsize=6.5, color=INK,
                        fontweight='bold' if m == 'bw' else 'normal')
    # colour wheel legend: hue = relative phase, saturation = phase consistency |c_j|
    cax = fig.add_axes([0.875, 0.365, 0.095, 0.27], projection='polar')
    th, r = np.meshgrid(np.linspace(-np.pi, np.pi, 181), np.linspace(0, 1, 30))
    rgb = cmap(plt.Normalize(-np.pi, np.pi)(th))[..., :3]
    rgb = r[..., None] * rgb + (1 - r[..., None]) * np.array(matplotlib.colors.to_rgb('#fcfcfb'))
    cax.pcolormesh(th, r, rgb[:-1, :-1], shading='flat', rasterized=True)
    cax.set_yticks([])
    cax.set_xticks([])
    x0, y0, rx, ry = 0.9225, 0.50, 0.062, 0.085
    for dx, dy, t in [(rx, 0, '0'), (0, ry, r'$\pi/2$'), (-rx, 0, r'$\pm\pi$'), (0, -ry, r'$-\pi/2$')]:
        fig.text(x0 + dx, y0 + dy, t, ha='center', va='center', fontsize=5, color=INK2)
    cax.grid(False)
    cax.spines['polar'].set_visible(False)
    fig.text(0.9225, 0.66, 'hue: phase\nfade: low\nconsistency', ha='center', va='bottom', fontsize=5.5, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def _fix_signs(F):
    """Resolve the column sign flips left by O(2): first column with positive sum, second column
    increasing from posterior to anterior (as in canonical_phase_map)."""
    F = F.copy()
    if F[:, 0].sum() < 0:
        F[:, 0] *= -1
    if F.shape[1] > 1 and np.polyfit(XY[:, 1], F[:, 1], 1)[0] < 0:
        F[:, 1] *= -1
    return F


def fig_eeg_columns(path=os.path.join(FIG, 'fig3_eeg_columns'), K=4):
    """EEG states shown as the columns of each method's own centroid: LEiDA u_1, the Grassmann frame
    [u_1, u_2] (top eigenvectors of the summed projections), and the BW factor M = [sqrt(l1) u_1, sqrt(l2) u_2]
    of the BW centroid, whose j-th row is the phasor (|c_j| cos phi_j, |c_j| sin phi_j)."""
    cache = os.path.join(RES, 'eoec_cache')
    d = np.load(os.path.join(RES, 'eoec.npz'))
    cols = {}
    for m in ('leida', 'grassmann'):
        z = np.load(os.path.join(cache, f'{m}_K{K}.npz'))
        rank = np.argsort(-z['fo'].mean((0, 1)))
        C = z['C'][rank]
        cols[m] = [_fix_signs(C[k][:, None] if C.ndim == 2 else C[k]) for k in range(K)]
    cols['bw'] = [_fix_signs(ig.ipl_eig(c)[2][0]) for c in d[f'bw_K{K}_maps']]
    s = json.load(open(os.path.join(RES, 'eoec_summary.json')))
    rows = [('leida', 0, 'LEiDA\n$\\mathbf{u}_1$'),
            ('grassmann', 0, 'Grassmann\n$\\mathbf{u}_1$'), ('grassmann', 1, 'Grassmann\n$\\mathbf{u}_2$'),
            ('bw', 0, 'BW\n$\\sqrt{\\lambda_1}\\mathbf{u}_1$'), ('bw', 1, 'BW\n$\\sqrt{\\lambda_2}\\mathbf{u}_2$')]
    unit_vmax = max(np.abs(np.concatenate(cols[m])).max() for m in ('leida', 'grassmann'))
    vmax = {'leida': unit_vmax, 'grassmann': unit_vmax, 'bw': 1.0}
    fig = plt.figure(figsize=(3.45, 4.0))
    gs = fig.add_gridspec(len(rows), K, left=0.12, right=0.84, top=0.95, bottom=0.02, hspace=0.30, wspace=0.04)
    for i, (m, j, lab) in enumerate(rows):
        occ = s['methods'][m][str(K)]['occupancy']
        for k in range(K):
            ax = fig.add_subplot(gs[i, k])
            topomap(ax, cols[m][k][:, j], vmax[m], cmap=PHASE_CMAP, title=f'{100 * occ[k]:.0f}%' if j == 0 else None)
            if k == 0:
                ax.text(-1.75, 0, lab, rotation=90, ha='center', va='center', fontsize=6, color=INK,
                        fontweight='bold' if m == 'bw' else 'normal')
    for (y0, h, v, t) in [(0.43, 0.48, unit_vmax, 'eigen-\nvector'), (0.04, 0.33, 1.0, 'entries\nof $\\mathbf{M}$')]:
        cax = fig.add_axes([0.87, y0 + 0.05, 0.02, h - 0.10])
        sm = plt.cm.ScalarMappable(cmap=PHASE_CMAP, norm=plt.Normalize(-v, v))
        cb = fig.colorbar(sm, cax=cax, ticks=[-v, 0, v] if v == 1 else [-0.2, -0.1, 0, 0.1, 0.2])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.5, length=1.5)
        cb.ax.set_title(t, fontsize=5.5, color=INK2, pad=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg_variants(K=4, path=None):
    """Exploration (not in the paper): K states of LEiDA, Grassmann and BW for eyes open / closed fitted
    separately and three EEG references. Titles: occupancy and mean lambda_1/P of the member samples."""
    path = path or os.path.join(FIG, f'figS5_eeg_variants_K{K}')
    d = np.load(os.path.join(RES, 'variants.npz'))
    s = json.load(open(os.path.join(RES, 'variants_summary.json')))
    conds, refs, methods = ('eo', 'ec'), ('native', 'average', 'laplacian'), ('leida', 'grassmann', 'bw')
    names = {'eo': 'eyes open', 'ec': 'eyes closed', 'leida': 'LEiDA', 'grassmann': 'Grassm.', 'bw': 'BW'}
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(len(refs) * len(methods), 2 * K + 1, width_ratios=[1] * K + [0.25] + [1] * K,
                          left=0.07, right=0.99, top=0.95, bottom=0.01, hspace=0.45, wspace=0.05)
    for a, c in enumerate(conds):
        fig.text(0.07 + 0.47 * a + 0.23, 0.975, names[c], ha='center', fontsize=8, color=INK, fontweight='bold')
        for b, r in enumerate(refs):
            for i, m in enumerate(methods):
                key = f'{c}_{r}_{m}_K{K}'
                occ, lam = s[key]['occupancy'], s[key]['lambda1_over_p']
                for k in range(K):
                    cm = d[f'{key}_maps'][k]
                    ax = fig.add_subplot(gs[b * len(methods) + i, a * (K + 1) + k])
                    topomap(ax, np.rad2deg(ig.canonical_phase_map(cm, XY)), 180, cmap=PHASE_SAT,
                            weight=2 * np.abs(cm) / np.abs(cm).max(), dots=False,
                            title=f'{100 * occ[k]:.0f}%  {lam[k]:.2f}')
                    if a == 0 and k == 0:
                        ax.text(-1.6, 0, names[m], rotation=90, ha='center', va='center', fontsize=6, color=INK,
                                fontweight='bold' if m == 'bw' else 'normal')
                        if i == 1:
                            ax.text(-2.5, 0, r, rotation=90, ha='center', va='center', fontsize=7, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def own_centroid_phasors(m, C):
    """Phasor reading of a method's own centroid: the factor F with F F^T the iPL matrix the centroid stands
    for, scaled so that ||F||^2 = P as for a single sample, with row j read as the complex number F_j1 + i F_j2.
    LEiDA: F = sqrt(P) u_1 (rank one, so phases are 0 or pi only). Grassmann: F = sqrt(P/2) [u_1, u_2]
    (the projector carries no eigenvalues, i.e. lambda_1 = lambda_2). BW: F = M, the centroid itself."""
    P = C.shape[-2] if C.ndim == 3 else C.shape[-1]
    if m == 'leida':
        return np.sqrt(P) * C.astype(complex)
    if m == 'grassmann':
        return np.sqrt(P / 2) * (C[..., 0] + 1j * C[..., 1])
    return C


def fig_eeg_own_centroids(path=os.path.join(FIG, 'fig3_eeg_own_centroids'), variant='ec_native', K=4,
                          rows=('leida', 'grassmann', 'bw'), cmap=None):
    """Each method's own centroid shown in one common format: the relative phases (and phasor lengths) of
    the factor F of the iPL matrix it represents. Titles: occupancy and mean lambda_1/P of member samples."""
    cmap = cmap or PHASE_RB
    v = np.load(os.path.join(RES, 'variants.npz'))
    vs = json.load(open(os.path.join(RES, 'variants_summary.json')))
    fig = plt.figure(figsize=(3.45, 2.5))
    gs = fig.add_gridspec(len(rows), K, left=0.09, right=0.835, top=0.93, bottom=0.02, hspace=0.30, wspace=0.04)
    labels = {'leida': 'LEiDA', 'grassmann': 'Grassmann', 'bw': 'BW'}
    for i, m in enumerate(rows):
        key = f'{variant}_{m}_K{K}'
        occ, lam = vs[key]['occupancy'], vs[key]['lambda1_over_p']
        F = own_centroid_phasors(m, v[f'{key}_maps'] if m == 'bw' else v[f'{key}_C'])
        for k in range(K):
            f = F[k]
            ax = fig.add_subplot(gs[i, k])
            topomap(ax, np.rad2deg(ig.canonical_phase_map(f, XY)), 180, cmap=cmap,
                    title=f'{100 * occ[k]:.0f}%  ({lam[k]:.2f})', weight=np.abs(f) / np.abs(f).max() * 2)
            if k == 0:
                ax.text(-1.5, 0, labels[m], rotation=90, ha='center', va='center', fontsize=6.5, color=INK,
                        fontweight='bold' if m == 'bw' else 'normal')
    cax = fig.add_axes([0.875, 0.365, 0.095, 0.27], projection='polar')
    th, r = np.meshgrid(np.linspace(-np.pi, np.pi, 181), np.linspace(0, 1, 30))
    rgb = cmap(plt.Normalize(-np.pi, np.pi)(th))[..., :3]
    rgb = r[..., None] * rgb + (1 - r[..., None]) * np.array(matplotlib.colors.to_rgb('#fcfcfb'))
    cax.pcolormesh(th, r, rgb[:-1, :-1], shading='flat', rasterized=True)
    cax.set_yticks([])
    cax.set_xticks([])
    cax.grid(False)
    cax.spines['polar'].set_visible(False)
    x0, y0, rx, ry = 0.9225, 0.50, 0.062, 0.085
    for dx, dy, t in [(rx, 0, '0'), (0, ry, r'$\pi/2$'), (-rx, 0, r'$\pm\pi$'), (0, -ry, r'$-\pi/2$')]:
        fig.text(x0 + dx, y0 + dy, t, ha='center', va='center', fontsize=5, color=INK2)
    fig.text(0.9225, 0.66, 'hue: phase\nfade: short\nphasor', ha='center', va='bottom', fontsize=5.5, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eoec_decoding(path=os.path.join(FIG, 'figS4_eoec_decoding')):
    """Supplementary (not in the paper): eyes open vs closed decoding from fractional occupancy."""
    s = json.load(open(os.path.join(RES, 'eoec_summary.json')))
    ks = [int(k) for k in s['methods']['bw']]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    for m in ORDER_PAPER + ['microstates']:
        acc = [s['methods'][m][str(k)]['accuracy'] for k in ks]
        if m == 'microstates':
            ax.plot(ks, acc, color=INK2, lw=1.0, ls='--', marker='x', ms=3.5, label='Microstates')
        else:
            label, col, mk = STYLE[m]
            ax.plot(ks, acc, color=col, lw=2.0 if m == 'bw' else 1.2, marker=mk, ms=3.2, mew=0.6,
                    mec='#fcfcfb', label=label)
    ax.axhline(0.5, color=AXIS, lw=0.8, ls=':')
    ax.set_xlabel('number of states K'); ax.set_ylabel('LOSO accuracy')
    ax.set_title('Eyes open vs closed from state occupancy (supplementary)', loc='left', color=INK)
    ax.set_ylim(0.45, 1.0); ax.set_xticks(ks)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc='lower right')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


# ------------------------------------------------------------------ supplementary (notebook)
def fig_synthetic_supplement(path=os.path.join(FIG, 'figS1_synthetic_supplement')):
    d = np.load(os.path.join(RES, 'synthetic.npz'))
    sig = d['sigmas']
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True)
    _lines(axes[0], sig, d, ['bw', 'bw_medoids', 'grassmann', 'complex'], 'main_', 'NMI')
    axes[0].set_title('main scenario: means vs medoids', loc='left')
    _lines(axes[1], sig, d, ORDER, 'shape_', '')
    axes[1].set_title('states differ only in shape', loc='left')
    _lines(axes[2], sig, d, ORDER, 'conjugate_', '')
    axes[2].set_title('mirror-image travelling waves', loc='left')
    for ax in axes:
        ax.set_xlabel(r'$\sigma$ (rad)')
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, fontsize=5.5, loc='lower left')
    handles, labels = axes[1].get_legend_handles_labels()
    axes[2].legend(handles, labels, frameon=False, fontsize=5.5, loc='center')
    axes[2].text(0.5, 0.06, 'all six cosine-iPL methods: NMI = 0', transform=axes[2].transAxes,
                 ha='center', fontsize=6, color=INK2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg_all_maps(path=os.path.join(FIG, 'figS2_eeg_all_methods'), K=4):
    d = np.load(os.path.join(RES, f'eeg{EEG_TAG}.npz'))
    s = json.load(open(os.path.join(RES, f'eeg_summary{EEG_TAG}.json')))
    methods = [m for m in ORDER]
    fig, axes = plt.subplots(len(methods), K, figsize=(0.9 * K + 0.5, 0.95 * len(methods)))
    maps = {m: np.stack([np.rad2deg(ig.canonical_phase_map(c, XY)) for c in d[f'{m}_K{K}_maps']]) for m in methods}
    vmax = 180
    for i, m in enumerate(methods):
        occ = s['methods'][m][str(K)]['occupancy']
        dwell = s['methods'][m][str(K)]['dwell_ms']
        for k in range(K):
            cons = np.abs(d[f'{m}_K{K}_maps'][k])
            topomap(axes[i, k], maps[m][k], vmax, cmap=PHASE_CYCLIC, title=f'{100 * occ[k]:.0f}%, {dwell[k]:.0f} ms',
                    weight=2 * cons / cons.max())
        axes[i, 0].text(-1.5, 0, STYLE[m][0], rotation=90, ha='center', va='center', fontsize=6)
    fig.suptitle(f'All methods, K = {K}: relative phase (±{vmax:.0f}°), occupancy, mean dwell time', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


def fig_eeg_consistency(path=os.path.join(FIG, 'figS3_bw_consistency'), K=4):
    d = np.load(os.path.join(RES, f'eeg{EEG_TAG}.npz'))
    fig, axes = plt.subplots(2, K, figsize=(0.95 * K, 2.0))
    C = d[f'bw_K{K}_maps']
    ph = np.stack([np.rad2deg(ig.canonical_phase_map(c, XY)) for c in C])
    vmax = 180
    cons = np.abs(C)
    seq = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
    for k in range(K):
        topomap(axes[0, k], ph[k], vmax, cmap=PHASE_CYCLIC, title='relative phase (cyclic)' if k == 0 else None)
        f_c = cons[k]
        topomap(axes[1, k], f_c - 0.5 * (f_c.max() + f_c.min()), 0.5 * np.ptp(f_c) + 1e-9, cmap=seq,
                title=f'|c| {f_c.min():.2f}–{f_c.max():.2f}')
    fig.suptitle('BW states (K = 4): relative phase and phase consistency |c_j|', fontsize=7)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    fig_methods()
    if os.path.exists(os.path.join(RES, 'synthetic.npz')):
        fig_synthetic()
        fig_synthetic_supplement()
    if os.path.exists(os.path.join(RES, 'eoec.npz')):
        fig_eeg_states()
        fig_eeg_states_saturated()
    if os.path.exists(os.path.join(RES, 'variants.npz')):
        fig_eeg_states_saturated(path=os.path.join(FIG, 'fig3_eeg'), variant='ec_native', cmap=PHASE_RB)
        fig_eeg_variants()
    if os.path.exists(os.path.join(RES, 'eoec.npz')):
        fig_eeg_columns()
        fig_eoec_decoding()
    if os.path.exists(os.path.join(RES, f'eeg{EEG_TAG}.npz')):
        fig_eeg()
        fig_eeg_all_maps()
        fig_eeg_consistency()
    print('figures written to', os.path.abspath(FIG))
