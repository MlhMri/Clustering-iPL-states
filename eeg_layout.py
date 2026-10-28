"""Approximate 2-D scalp layout of the 64-channel PhysioNet EEGMMIDB montage (10-10 system).

Azimuthal projection: Cz at the origin, nose at +y, left hemisphere at -x. Midline electrodes
are spaced 0.2 per 10% step (Fpz at 0.8, Oz at -0.8, Iz at -1.0); the lateral end of every
coronal row lies on the circle r = 0.8 at the standard 10-10 azimuth (Fp1 18 deg, AF7 36 deg,
F7 54 deg, FT7 72 deg, T7 90 deg, ...). Positions are approximate and meant for display and
for defining smooth synthetic phase patterns only.
"""
import numpy as np

LABELS = ('Fc5 Fc3 Fc1 Fcz Fc2 Fc4 Fc6 C5 C3 C1 Cz C2 C4 C6 Cp5 Cp3 Cp1 Cpz Cp2 Cp4 Cp6 '
          'Fp1 Fpz Fp2 Af7 Af3 Afz Af4 Af8 F7 F5 F3 F1 Fz F2 F4 F6 F8 Ft7 Ft8 T7 T8 T9 T10 '
          'Tp7 Tp8 P7 P5 P3 P1 Pz P2 P4 P6 P8 Po7 Po3 Poz Po4 Po8 O1 Oz O2 Iz').split()

_ROWS = {'Fp': (0.8, 18), 'Af': (0.6, 36), 'F': (0.4, 54), 'Fc': (0.2, 72), 'C': (0.0, 90),
         'Cp': (-0.2, 108), 'P': (-0.4, 126), 'Po': (-0.6, 144), 'O': (-0.8, 162)}
_ALIAS = {'T7': 'C7', 'T8': 'C8', 'Ft7': 'Fc7', 'Ft8': 'Fc8', 'Tp7': 'Cp7', 'Tp8': 'Cp8'}


def _position(label):
    fixed = {'T9': (-1.0, 0.0), 'T10': (1.0, 0.0), 'Iz': (0.0, -1.0)}
    if label in fixed:
        return fixed[label]
    name = _ALIAS.get(label, label)
    row = name.rstrip('z0123456789')
    suffix = name[len(row):]
    y0, azimuth = _ROWS[row]
    if suffix == 'z':
        return (0.0, y0)
    k = int(suffix)
    side = -1.0 if k % 2 else 1.0
    frac = 1.0 if row in ('Fp', 'O') else ((k + 1) // 2) / 4
    end = 0.8 * np.array([np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth))])
    pos = (1 - frac) * np.array([0.0, y0]) + frac * end
    return (side * pos[0], pos[1])


XY = np.array([_position(lab) for lab in LABELS])
