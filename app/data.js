// All numbers in this file are extracted from real saved artifacts in outputs/.
// Sources are noted inline. Do not invent. If you change a number here, also
// re-verify against the source file.

window.NOVONUS = {

  stage1: {
    // outputs/stage1/meta.json + summary.txt
    subject: "DB2_s1",
    fs_hz: 2000,
    emg_channels: 12,
    acc_channels: 36,
    glove_channels: 22,
    total_samples: 4361620,
    duration_min: 36.35,
    blocks: "E1 + E2",
    dataset: "Ninapro DB2",
    video: "assets/stage1/oscilloscope_preview.mp4",
    still: "assets/stage1/segment_raw_vs_clean.png",
    poster: "assets/stage1/oscilloscope_preview_frame.png",
  },

  stage2: {
    // outputs/stage2/training_summary.txt
    best_val_intent_acc_pct: 94.18,
    best_val_force_mse: 0.000353,
    best_epoch: 14,
    epochs_run: 24,
    n_intent_classes: 5,
    intent_names: ["REST", "REACHING", "GRIPPING", "STABILIZING", "RELEASING"],
    loss_curve: "assets/stage2/loss_curve.png",
    confusion_matrix: "assets/stage2/confusion_matrix.png",
    video: "assets/stage2/intent_and_force_preview.mp4",

    // DSP chain - from outputs/stage1/meta.json:
    //   "HP-Notch-BP-Rect-RMS100ms-MVCnorm"
    dsp_steps: [
      { name: "High-pass filter", detail: "remove DC drift / motion artifact" },
      { name: "Notch filter", detail: "kill 50/60 Hz line noise" },
      { name: "Band-pass filter", detail: "EMG band 20-450 Hz" },
      { name: "Rectify", detail: "absolute value" },
      { name: "RMS envelope (100 ms)", detail: "smoothed activation envelope" },
      { name: "MVC normalize", detail: "per-channel 99th-pct calibration" },
    ],
  },

  stage2_force: {
    // outputs/stage2/force_model_e3/force_accuracy_summary.txt
    // Envelope-only baseline E3 force model (the model behind force_prediction_e3 + force_scatter_e3)
    r_squared: 0.9596,
    pearson_r: 0.9798,
    rmse_norm: 0.0602,
    rmse_real_N: 1.1473,
    mae_real_N: 0.7339,
    mape_pct: 9.77,
    test_samples: 6982,
    prediction_png: "assets/stage2/force_prediction_e3.png",
    scatter_png: "assets/stage2/force_scatter_e3.png",
    crush_threshold_N: 6.0,
    notes_caption:
      "EMG-predicted grip force vs real 6-axis force sensors, held-out test (repetition 6 of E3, never seen during training). Subject DB2_s1.",
  },

  stage3: {
    // outputs/stage3/metrics.json + validation.json
    scene_render: "assets/stage3/scene_render.png",
    video: "assets/stage3/synced_demo.mp4",
    idle_video: "assets/stage3/idle_motion.mp4",
    max_contact_N: 1.237,
    crush_threshold_N: 6.0,
    ever_exceeded_crush: false,
    grip_gauge_max: 0.306,
    grip_gauge_mean: 0.248,
    grape_in_target_zone: true,
    grape_xy_err_mm: 27.2,
    simulator: "MuJoCo Warp (NVIDIA Newton solver)",
    arm: "UR5e",
    target: "grape",
  },

  stage4: {
    // outputs/stage4/dataset_summary.txt
    n_demos: 30,
    avg_duration_s: 14.01,
    total_timesteps: 12360,
    applied_max_N: 3.80,
    contact_max_N: 1.44,
    crush_threshold_N: 6.0,
    all_below_crush: true,
    xy_err_max_mm: 20.8,
    xy_err_mean_mm: 20.4,
    in_zone: true,
    safe_band: "2.0 - 4.0 N",
    grid_png: "assets/stage4/demo_grid.png",
    video: "assets/stage4/demo_sample.mp4",
  },

  stage5: {
    // outputs/stage5/augmentation_stats.json
    n_baselines: 30,
    raw_scenarios: 123,
    passing: 100,
    pass_rate_pct: 81.3,
    rejection_breakdown: {
      PATH_DEVIATION: 0,
      CONTACT_TIMING: 0,
      TASK_FAILURE: 0,
      FORCE_OUT_OF_BAND: 0,
      FORCE_PROFILE_MISMATCH: 23,
      FORCE_OVERSHOOT: 0,
    },
    peak_force_mean_N: 1.371,
    peak_force_max_N: 2.045,
    crush_threshold_N: 6.0,
    all_below_crush: true,
    grid_png: "assets/stage5/augmentation_grid.png",
    scenario_video: "assets/stage5/scenario_00000.mp4",
    verify_checks: [
      // 6-point verification, per spec
      { key: "path_similarity", label: "Path similarity vs baseline" },
      { key: "contact_timing", label: "Contact timing within window" },
      { key: "task_success", label: "Task success (grape placed)" },
      { key: "peak_force", label: "Peak force within band" },
      { key: "force_profile", label: "Force profile matches baseline" },
      { key: "force_overshoot", label: "No overshoot above crush" },
    ],
  },

  stage7: {
    // outputs/stage7/README.md + loss_history_emg.json
    best_val_mse: 0.0066,
    best_epoch: 164,
    stopped_epoch: 184,
    train_mse_at_stop: 0.0025,
    wall_time: "5h 17min",
    trainable_params_M: 76.76,
    fusion_dim: 960,
    n_demos_total: 130,
    train_samples: 45320,
    val_samples: 8240,
    loss_curve: "assets/stage7/loss_curve_emg.png",
    loss_history_json: "assets/stage7/loss_history_emg.json",
    diag_video: "assets/stage7/diag_replay.mp4",
    exec_video: "assets/stage7/execution_trial_02.mp4",
    // honest result on closed-loop:
    closed_loop_success: 0,
    closed_loop_trials: 5,
    closed_loop_peak_max_N: 1.11,
    crush_threshold_N: 6.0,
    force_safe_in_all_trials: true,
  },
};
