"""Stage 3 entry point.

End-to-end:
  1. Build the canonical scene XML (data/assets/scene.xml).
  2. Initialise the MuJoCo Warp GPU backend on cuda:0.
  3. Render the static scene PNG + idle motion MP4.
  4. Run validation (idle stability + reach-grip-stabilize-release cycle).
  5. Render the synchronised demo MP4 (sim + EMG + force gauge).
  6. Write the README.

Run with:  python -m src.stage3_sim.run_stage3
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import physics_constants as pc
from . import scene as scene_mod
from .render_utils import render_idle_motion_mp4, render_scene_png
from .validate import run_validation
from .visualize_sync import render_synced_demo
from .warp_backend import WarpBackend, save_backend_report


README = """# Stage 3 outputs — MuJoCo Warp simulation driven by Stage 2 LSTM

## What this stage built

A physically-simulated UR5e tabletop scene running on the MuJoCo Warp
(Newton) GPU backend, with the arm's motion phases driven by the Stage 2
LSTM's predicted intent class and the grip force during contact derived
from the Stage 2 LSTM force-head output, scaled into a safe band well
below the grape's 6 N crush threshold.

## Files

| file | what it is |
|------|------------|
| `scene_render.png`         | static 640x480 render of the scene from the fixed 45 deg camera |
| `idle_motion.mp4`          | 7 s clip of a gentle idle hover — sanity check that physics + rendering work |
| `synced_demo.mp4`          | ~21 s synchronised demo: sim left, EMG scope + intent label + force gauge right |
| `validation.json`          | machine-readable pass/fail for each cycle phase + grip-force ceiling check |
| `phase_log.csv`            | per-tick log: time, phase, gauge, target grip force, measured contact force |
| `backend.txt` / `.json`    | Warp backend report (cuda device, solver, integrator, model sizes) |
| `metrics.json`             | per-phase max gauge value, max grip-force, ever-exceeded-crush flag |
| `README.md`                | this file |

## Single canonical scene XML

`data/assets/scene.xml` is the authoritative environment file for every
subsequent stage. Stage 4+ should load it via:

```python
from src.stage3_sim.warp_backend import WarpBackend
backend = WarpBackend()         # builds data/assets/scene.xml on first run
```

## Grape fragility envelope

All physics constants live in `src/stage3_sim/physics_constants.py`:

- grape mass     : 5 g
- grape radius   : 2 cm
- crush threshold: **6 N** (named constant `GRAPE_CRUSH_THRESHOLD_N`)
- safe grip band : **2-4 N** (named constants `GRAPE_SAFE_GRIP_{MIN,MAX}_N`)
- gauge scale    : 0..10 N full scale (so the safe band reads **0.2-0.4** on
  the gauge and the **6 N crush line lands at 0.6**, clearly above the band)
- timestep       : 2 ms (500 Hz physics)
- camera         : `scene_cam` at 45 deg above-front, 640x480 (DINOv2-ready)

Importing the constants anywhere else keeps the demo consistent — if we
ever want a stiffer object we change it here and the gauge, the actuator
cap, and the visualization all follow.

## How the LSTM drives the arm

`src/stage3_sim/emg_sync.py` walks through a scripted phase order
(REST -> REACHING -> GRIPPING -> STABILIZING -> RELEASING -> REST), but
the EMG signal for each phase is sourced from a **real Stage 1 validation
sub-segment** where the trained LSTM in fact predicts that intent class.
For every tick the streamer runs a real forward pass of the LSTM over real
EMG, so the EMG oscilloscope shows the same signal the model is reading
and the gauge value is the model's force-head output — they are not
faked.

During GRIPPING / STABILIZING, the LSTM's sigmoid in [0, 1] is scaled into
the safe grape grip band [2, 4] N. The position-controlled jaws have a
`forcerange` capped at the crush threshold, so the simulated contact
force on the grape cannot exceed 6 N even under control divergence.

## UR5e source

The UR5e model is downloaded from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
- `universal_robots_ur5e/`. The XML and 20 OBJ meshes are placed under
`data/assets/ur5e/`. We extended that XML in-place at scene-build time to
add the table, the grape, a target zone, a 45 deg camera, and a two-jaw
parallel gripper bolted to the wrist_3 attachment_site.
"""


def main() -> int:
    out_dir = pc.STAGE3_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== Stage 3 — build scene ===")
    scene_path = scene_mod.build_scene_xml()
    print(f"  scene XML -> {scene_path}")

    print("\n=== Stage 3 — initialise MuJoCo Warp GPU backend ===")
    backend = WarpBackend()
    print(backend.info.report())
    save_backend_report(backend, out_dir)

    print("\n=== Stage 3 — static scene PNG ===")
    png_path = render_scene_png(backend, out_dir / "scene_render.png")
    print(f"  wrote {png_path}")

    print("\n=== Stage 3 — idle motion MP4 ===")
    backend.reset()
    idle_path = render_idle_motion_mp4(backend, out_dir / "idle_motion.mp4",
                                       duration_s=7.0, fps=30)
    print(f"  wrote {idle_path}")

    print("\n=== Stage 3 — validation ===")
    res = run_validation(out_dir=out_dir)

    print("\n=== Stage 3 — synchronised demo MP4 ===")
    metrics = render_synced_demo(
        out_mp4=out_dir / "synced_demo.mp4",
        write_phase_log_to=out_dir / "synced_phase_log.csv",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    print(f"  per-phase max gauge: {metrics['max_gauge_by_phase']}")
    print(f"  per-phase max target force (N): {metrics['max_force_N_by_phase']}")
    print(f"  max measured contact (N): {metrics['max_contact_force_N']:.2f}  "
          f"(crush = {pc.GRAPE_CRUSH_THRESHOLD_N:.0f} N)")
    print(f"  ever exceeded crush: {metrics['ever_exceeded_crush']}")

    print("\n=== Stage 3 — README ===")
    (out_dir / "README.md").write_text(README, encoding="utf-8")
    print(f"  wrote {out_dir/'README.md'}")

    elapsed = time.time() - t0
    print(f"\n[stage3] DONE in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
