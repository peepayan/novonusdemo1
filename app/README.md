# Novonus — Force-Intelligence Pipeline (demo web app)

A single-page demo app that walks through Stages 1–7 of the Novonus pipeline
as a guided, app-like presentation over the real saved artifacts in
`outputs/stage1/ … outputs/stage7/`.

This is **not live inference**. It is a presentational replay/visualization
of the real saved results, designed for screen-recording and investor demos.
All displayed metrics are extracted from the real saved files (no
hardcoded guesses). See `data.js` for source-file traceability.

## Run

From the repo root (`novonusdemo1/`):

```bash
python -m http.server --directory app 8000
```

Then open <http://localhost:8000> in any modern browser.

You can also just double-click `app/index.html` — but browsers will block
inline `<video>` playback for `file://`, so the local server is the
preferred mode for demos.

## Navigation

- Click any stage in the top flow bar to jump to it.
- Use **← / →** arrow keys, or the **Prev / Next** buttons in the footer.
- Each stage screen has a primary action button (e.g. "Gather EMG Data",
  "Run DSP Pipeline") that triggers a short loading animation, then reveals
  the real artifact and animated visualizations for that stage.

## Stage map

| # | Title | Real artifacts shown |
|---|---|---|
| 1 | Capture EMG | `oscilloscope_preview.mp4`, `segment_raw_vs_clean.png` |
| 2 | Signal Processing (DSP) | filter-chain checklist, `segment_raw_vs_clean.png` |
| 3 | Intent + Force Model | `loss_curve.png` replay, `intent_and_force_preview.mp4`, `confusion_matrix.png` |
| 4 | Force Validation | `force_prediction_e3.png`, `force_scatter_e3.png` — R² = 0.96 vs real force sensors |
| 5 | Simulation Scene | `scene_render.png`, `synced_demo.mp4`, crush-threshold bar |
| 6 | Augmentation + Verification | `augmentation_grid.png`, animated tile + 6-check verification, `scenario_00000.mp4` |
| 7 | Policy Training | `loss_curve_emg.png` replay, architecture diagram, diffusion-denoising concept, `diag_replay.mp4` |
| — | Summary | Branded close screen with key validated metrics |

## Honesty constraints (enforced in code)

- All metrics are pulled from real files (see `data.js`).
- All videos/plots are the real saved artifacts.
- The Stage 7 closed-loop result is presented honestly: training converged,
  force-safety held in every trial, but task completion was 0/5 — closed-loop
  autonomous execution is in progress pending more demonstration diversity.
- "Replay" animations (loss curves, scenario tiles, verification, denoising
  concept) are clearly labelled as presentational replays/illustrations.
