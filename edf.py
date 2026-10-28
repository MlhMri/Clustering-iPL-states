"""Minimal EDF reader (16-bit European Data Format, as used by PhysioNet EEGMMIDB)."""
import numpy as np


def read_edf(path):
    """Return (labels, fs, X) with X of shape (n_signals, n_samples) in physical units.
    The EDF Annotations channel is dropped."""
    with open(path, 'rb') as f:
        h = f.read(256)
        n_bytes, n_rec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])
        sh = f.read(n_bytes - 256)

        def field(offset, width):
            return [sh[offset + i * width: offset + (i + 1) * width].decode('latin-1').strip() for i in range(ns)]

        o = 0
        labels = field(o, 16); o += 16 * ns
        o += 80 * ns + 8 * ns                       # transducer, physical dimension
        pmin = np.array(field(o, 8), float); o += 8 * ns
        pmax = np.array(field(o, 8), float); o += 8 * ns
        dmin = np.array(field(o, 8), float); o += 8 * ns
        dmax = np.array(field(o, 8), float); o += 8 * ns
        o += 80 * ns                                # prefiltering
        nsamp = np.array(field(o, 8), int)
        raw = np.frombuffer(f.read(), dtype='<i2')
    rec_len = nsamp.sum()
    n_rec = len(raw) // rec_len if n_rec < 0 else min(n_rec, len(raw) // rec_len)
    raw = raw[:n_rec * rec_len].reshape(n_rec, rec_len)
    keep = [i for i, lab in enumerate(labels) if 'annotation' not in lab.lower()]
    starts = np.concatenate([[0], np.cumsum(nsamp)])
    gain = (pmax - pmin) / (dmax - dmin)
    X = np.stack([(raw[:, starts[i]:starts[i + 1]].ravel() - dmin[i]) * gain[i] + pmin[i] for i in keep])
    return [labels[i].rstrip('.') for i in keep], nsamp[keep[0]] / dur, X
