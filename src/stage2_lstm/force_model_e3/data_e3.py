"""Load DB2 Subject 1's E3 block (S1_E3_A1.mat) — the only DB2 block that
contains *measured* force-sensor data. Used here as the training source for
a dedicated EMG-to-force regressor (no glove channel; not used elsewhere)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio


FS_HZ: float = 2000.0  # DB2 spec


@dataclass
class E3Block:
    emg: np.ndarray         # (T, 12)  raw EMG voltage
    acc: np.ndarray         # (T, 36)  Delsys IMU triaxial accel x 12 sensors
    force: np.ndarray       # (T, 6)   measured force per channel (calibrated)
    forcecal: np.ndarray    # (2, 6)   calibration min/max per channel (units N-ish)
    restimulus: np.ndarray  # (T,)     int16, movement label 0/41..49 (force tasks)
    rerepetition: np.ndarray# (T,)     int16, 0=gap, 1..6=rep index
    fs: float = FS_HZ

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def report(self) -> str:
        lines = []
        lines.append("E3 raw block structure (S1_E3_A1.mat)")
        lines.append("-" * 50)
        lines.append(f"  sampling rate    : {self.fs:.0f} Hz (DB2 spec)")
        lines.append(f"  total samples    : {self.n_samples:,}")
        lines.append(f"  duration         : {self.duration_s/60:.2f} min")
        lines.append(f"  emg              : shape={self.emg.shape}  dtype={self.emg.dtype}")
        lines.append(f"  acc              : shape={self.acc.shape}  dtype={self.acc.dtype}")
        lines.append(f"  force            : shape={self.force.shape}  dtype={self.force.dtype}")
        lines.append(f"  forcecal         : shape={self.forcecal.shape}  dtype={self.forcecal.dtype}")
        lines.append(f"  restimulus       : shape={self.restimulus.shape}")
        lines.append(f"  rerepetition     : shape={self.rerepetition.shape}")
        lines.append("")
        lines.append("  restimulus labels present: "
                     + str(sorted(np.unique(self.restimulus).tolist())))
        lines.append("  reps                     : "
                     + str(sorted(np.unique(self.rerepetition).tolist())))
        lines.append("")
        lines.append("  per-channel measured force (calibrated units):")
        for c in range(self.force.shape[1]):
            x = self.force[:, c]
            lines.append(
                f"    ch{c}: min={x.min():+7.3f}  max={x.max():+7.3f}  "
                f"mean={x.mean():+7.3f}  std={x.std():.3f}"
            )
        lines.append("")
        lines.append("  forcecal (per-channel calibration constants, 2 x 6):")
        for row in self.forcecal:
            lines.append("    " + "  ".join(f"{v:+7.3f}" for v in row))
        lines.append("")
        lines.append(
            "  -> chose to predict the L2 magnitude across the 6 force channels\n"
            "     (overall force effort) as a single scalar regression target,\n"
            "     because the channels measure different finger-axis forces and a\n"
            "     magnitude summary aligns with the 'one number per timestep'\n"
            "     interface used by the rest of the pipeline."
        )
        return "\n".join(lines)


def load_e3(mat_path: str | Path) -> E3Block:
    p = Path(mat_path)
    m = sio.loadmat(p, squeeze_me=False, struct_as_record=False)
    needed = ["emg", "acc", "force", "forcecal", "restimulus", "rerepetition"]
    missing = [k for k in needed if k not in m]
    if missing:
        raise KeyError(f"{p.name}: missing keys {missing}")
    emg = m["emg"].astype(np.float32, copy=False)
    acc = m["acc"].astype(np.float32, copy=False)
    force = m["force"].astype(np.float32, copy=False)
    forcecal = m["forcecal"].astype(np.float32, copy=False)
    restim = m["restimulus"].astype(np.int16, copy=False).ravel()
    rerep = m["rerepetition"].astype(np.int16, copy=False).ravel()
    # Crop to common length (restimulus/rerepetition are 1 sample shorter)
    n = min(emg.shape[0], acc.shape[0], force.shape[0],
            restim.shape[0], rerep.shape[0])
    return E3Block(
        emg=emg[:n], acc=acc[:n], force=force[:n],
        forcecal=forcecal,
        restimulus=restim[:n], rerepetition=rerep[:n],
    )
