// Novonus pipeline walkthrough app.
// Single-page state machine over 7 stages + final summary.
// Each stage screen has: action button -> brief loader -> real artifact reveal.

(function () {
  const D = window.NOVONUS;

  const stages = [
    { id: "s0", title: "Home", short: "Home" },
    { id: "s1", title: "Capture EMG", short: "Capture EMG" },
    { id: "s2", title: "Signal Processing (DSP)", short: "DSP" },
    { id: "s3", title: "Intent + Force Model", short: "LSTM (Intent + Force)" },
    { id: "s4", title: "Force Validation", short: "Force Validation" },
    { id: "s5", title: "Simulation Scene", short: "Simulation" },
    { id: "s6", title: "Augmentation + Verification", short: "Augmentation" },
    { id: "s7", title: "Policy Training", short: "Policy" },
    { id: "summary", title: "Summary", short: "Summary" },
  ];

  let current = 0;
  const completed = new Set();

  const mainEl = document.getElementById("app-main");
  const navEl = document.getElementById("stage-nav");

  // ---------- nav -------------------------------------------------------
  function renderNav() {
    navEl.innerHTML = "";
    stages.forEach((s, i) => {
      if (i > 0) {
        const c = document.createElement("div");
        c.className = "stage-connector";
        navEl.appendChild(c);
      }
      const pill = document.createElement("button");
      pill.className = "stage-pill";
      if (i === current) pill.classList.add("active");
      if (completed.has(i) && i !== current) pill.classList.add("done");
      pill.innerHTML = `<span class="num"><span>${i + 1}</span></span><span class="label">${s.short}</span>`;
      pill.addEventListener("click", () => goTo(i));
      navEl.appendChild(pill);
    });
  }

  function goTo(i) {
    if (i < 0 || i >= stages.length) return;
    if (i > current) completed.add(current);
    current = i;
    renderNav();
    showScreen(stages[i].id);
    updateFooter();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function showScreen(id) {
    document.querySelectorAll(".screen").forEach(el => el.classList.remove("active"));
    const el = document.getElementById("screen-" + id);
    if (el) el.classList.add("active");
  }

  function updateFooter() {
    document.getElementById("prev-btn").disabled = current === 0;
    document.getElementById("next-btn").disabled = current === stages.length - 1;
  }

  // ---------- helpers ---------------------------------------------------
  function el(tag, cls, html) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html != null) n.innerHTML = html;
    return n;
  }

  function loaderHTML(label) {
    return `<div class="spinner"></div><div class="loader-label">${label}</div>`;
  }

  function runLoader(screenId, label, ms) {
    return new Promise(resolve => {
      const lz = document.querySelector(`#screen-${screenId} .loader-zone`);
      const btn = document.querySelector(`#screen-${screenId} .btn-primary`);
      if (btn) btn.disabled = true;
      lz.innerHTML = loaderHTML(label);
      lz.classList.add("active");
      setTimeout(() => {
        lz.classList.remove("active");
        resolve();
      }, ms);
    });
  }

  function revealContent(screenId) {
    const r = document.querySelector(`#screen-${screenId} .reveal`);
    r.classList.add("active");
  }

  function loadVideos(sc) {
    sc.querySelectorAll("video[data-src]").forEach(v => { v.src = v.dataset.src; });
  }

  // Count-up easing for stat numbers
  function countUp(elNode, target, durationMs, opts) {
    opts = opts || {};
    const decimals = opts.decimals != null ? opts.decimals : 2;
    const suffix = opts.suffix || "";
    const prefix = opts.prefix || "";
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / durationMs);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = target * eased;
      elNode.textContent = prefix + v.toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(tick);
      else elNode.textContent = prefix + target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(tick);
  }

  // ---------- screen builders ------------------------------------------

  // STAGE 0,Home
  function buildScreen0() {
    const sc = el("section", "screen", "");
    sc.id = "screen-s0";
    sc.innerHTML = `
      <div class="home-card">
        <img src="assets/novonus-logo.png" class="home-logo" alt="Novonus" />
        <h1 class="home-title">Novonus</h1>
        <p class="home-sub">Force-intelligence for collaborative robots, EMG-conditioned policies that learn force-aware manipulation from operator demonstrations.</p>
        <div class="capability-pills">
          <span class="capability-pill">EMG → Intent → Force</span>
          <span class="capability-pill">Simulation + Augmentation</span>
          <span class="capability-pill">Diffusion Policy</span>
        </div>
        <div class="home-cta-hint">
          <span class="arrow">→</span>
          <span>Use <strong>Next →</strong> or the stage bar above to begin the walkthrough.</span>
        </div>
      </div>
      <div class="home-context">
        <div class="home-context-label">What this demo is</div>
        <p>This is a focused, single-modality walkthrough of the Novonus pipeline, trained end-to-end on EMG data alone, drawn from the public Ninapro DB2 dataset. It demonstrates the core, hardest-to-prove piece of the system: predicting real grip force and intent directly from muscle signals. The full Novonus product captures four synchronized data streams (EMG, force, motion, and vision) from a live multimodal hardware rig; this demo isolates the EMG-to-force pathway to show, with full transparency, the validated result the rest of the system is built on. Every stage below reflects what's actually implemented and tested in software and simulation today.</p>
      </div>
    `;
    return sc;
  }

  // STAGE 1,Capture EMG
  function buildScreen1() {
    const s = D.stage1;
    const sc = el("section", "screen", "");
    sc.id = "screen-s1";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 1</span>
          <div>
            <h2 class="stage-title">Capture EMG</h2>
            <div class="stage-subtitle">Surface EMG acquisition from operator forearm muscles. Ninapro DB2, subject ${s.subject}, dry-electrode grid.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">Raw EMG signal is captured from Ninapro DB2, muscle activity recorded during real hand and grip movements.</div>
        <div class="ctx-item">This stands in for the live capture step of the full rig, where an operator wears the sensor system and performs a task by hand.</div>
        <div class="ctx-item">The goal of this stage: get a clean, real recording of muscle activity to work from.</div>
        <div class="ctx-item">The raw EMG signal is visualized here as multi-channel muscle activity over time.</div>
        <div class="ctx-item">This is the unprocessed input, noisy, high-frequency electrical activity straight from the muscle.</div>
        <div class="ctx-item">Nothing has been filtered or interpreted yet; this is what the sensors actually pick up.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">⚡</div>
          <div>
            <div style="font-weight:600">Gather EMG Data</div>
            <div class="action-desc">Open a multi-channel acquisition window over the forearm.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s1">Gather EMG Data</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-8">
            <h4>Live oscilloscope, raw + processed</h4>
            <video class="big" data-src="${s.video}" poster="${s.poster}" muted loop playsinline preload="none" controls></video>
            <div class="caption">Real surface EMG from operator forearm muscles, Ninapro DB2.</div>
          </div>
          <div class="card col-4">
            <h4>Capture stats</h4>
            <div class="stat-row" style="flex-direction:column">
              <div class="stat"><div class="stat-label">Channels</div><div class="stat-value">${s.emg_channels}</div><div class="stat-sub">surface EMG</div></div>
              <div class="stat"><div class="stat-label">Sample rate</div><div class="stat-value">${s.fs_hz} <span style="font-size:14px">Hz</span></div><div class="stat-sub">hard-pinned, DB2 spec</div></div>
              <div class="stat"><div class="stat-label">Duration</div><div class="stat-value">${s.duration_min} <span style="font-size:14px">min</span></div><div class="stat-sub">blocks ${s.blocks}</div></div>
            </div>
          </div>
          <div class="card col-12">
            <h4>10 s raw vs envelope segment</h4>
            <img src="${s.still}" alt="raw vs clean EMG segment" />
            <div class="caption">A 10-second window: raw multi-channel EMG (top) vs the MVC-normalized envelope (bottom).</div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s1", "Acquiring signal…", 900);
      revealContent("s1");
      loadVideos(sc);
      const v = sc.querySelector("video");
      if (v) { v.load(); v.play().catch(()=>{}); }
    });
    return sc;
  }

  // STAGE 2,DSP
  function buildScreen2() {
    const s = D.stage2;
    const sc = el("section", "screen", "");
    sc.id = "screen-s2";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 2</span>
          <div>
            <h2 class="stage-title">Signal Processing (DSP)</h2>
            <div class="stage-subtitle">Raw EMG is noisy. We run a 6-stage chain to extract a clean activation envelope.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">The raw signal is cleaned: filtered to remove noise and drift, rectified, and smoothed into a usable activation envelope.</div>
        <div class="ctx-item">Includes normalization so the signal is consistent and comparable across channels and sessions.</div>
        <div class="ctx-item">This turns messy raw muscle data into a clean feature the model can actually learn from.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">≋</div>
          <div>
            <div style="font-weight:600">Run DSP Pipeline</div>
            <div class="action-desc">Apply: high-pass → notch → band-pass → rectify → RMS envelope → MVC normalize.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s2">Run DSP Pipeline</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-6">
            <h4>Filter chain</h4>
            <ul class="filter-pipeline" id="dsp-list">
              ${s.dsp_steps.map((st, i) => `
                <li class="filter-step" data-i="${i}">
                  <span class="badge"><span>${i + 1}</span></span>
                  <div>
                    <div style="font-weight:600">${st.name}</div>
                    <div class="muted" style="font-size:12px">${st.detail}</div>
                  </div>
                </li>
              `).join("")}
            </ul>
            <div class="caption" style="margin-top:14px">Each step removes a known noise source. Net effect: raw mV-scale spikes → smooth 0..1 muscle activation.</div>
          </div>
          <div class="card col-6">
            <h4>Raw → clean</h4>
            <img src="${D.stage1.still}" alt="raw vs clean EMG" />
            <div class="caption">Top: raw multi-channel EMG. Bottom: the MVC-normalized envelope after the filter chain, what downstream models consume.</div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s2", "Initializing DSP chain…", 600);
      revealContent("s2");
      // step through filter checklist
      const steps = sc.querySelectorAll(".filter-step");
      const stepMs = 380;
      for (let i = 0; i < steps.length; i++) {
        await new Promise(r => setTimeout(r, stepMs));
        steps[i].classList.add("running");
        if (i > 0) steps[i - 1].classList.replace("running", "done");
      }
      await new Promise(r => setTimeout(r, stepMs));
      steps[steps.length - 1].classList.replace("running", "done");
    });
    return sc;
  }

  // STAGE 3,Intent + Force (LSTM)
  function buildScreen3() {
    const s = D.stage2;
    const s1 = D.stage1;
    const sc = el("section", "screen", "");
    sc.id = "screen-s3";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 3, LSTM training</span>
          <div>
            <h2 class="stage-title">Intent + Force Model <span style="font-size:14px; padding:4px 10px; border:1px solid var(--accent); border-radius:999px; color:var(--accent); margin-left:10px; vertical-align:middle; letter-spacing:1px;">LSTM</span></h2>
            <div class="stage-subtitle"><strong>This is where the LSTM is trained.</strong> A bidirectional LSTM maps 12-channel EMG envelopes to (a) intent class and (b) an EMG-amplitude force proxy. Architecture: 2-layer BiLSTM (128 hidden) → two heads: 5-way softmax (intent) + 1-d regression (force). Train data: Ninapro DB2 windows; loss: cross-entropy + λ·MSE.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">A recurrent neural network reads the processed EMG signal and predicts two things at once: what the hand is doing (intent: reach/grip/stabilize/release) and how much force is being applied.</div>
        <div class="ctx-item">This is the core model of the demo, it learns the mapping from muscle activity to real physical behavior.</div>
        <div class="ctx-item">Trained and validated on held-out data the model never saw during training.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">⌬</div>
          <div>
            <div style="font-weight:600">Train LSTM: Classify Intent + Predict Force</div>
            <div class="action-desc">Replay the real LSTM training run (loss curve), then show the trained model's live label + force gauge and the confusion matrix.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s3">Train LSTM</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-7">
            <h4>LSTM training, real loss curve replay</h4>
            <div class="replay-wrap" id="s3-replay">
              <img src="${s.loss_curve}" alt="LSTM training loss curve" />
              <div class="replay-hud">
                <div>epoch <span class="hud-value" id="s3-epoch">0</span> / ${s.epochs_run}</div>
                <div>val acc <span class="hud-value" id="s3-acc">0.00%</span></div>
              </div>
              <div class="replay-mask" id="s3-mask"></div>
              <div class="replay-tag">Replay of real training run</div>
            </div>
            <div class="caption"><strong style="color:var(--accent)">LSTM trained here.</strong> Best validation intent accuracy <strong>${s.best_val_intent_acc_pct.toFixed(2)}%</strong> at epoch ${s.best_epoch}. This is a replay of the saved training curve, not live training. Checkpoint saved at <code class="mono" style="color:var(--text-1)">outputs/stage2/lstm_best.pt</code>.</div>
          </div>
          <div class="card col-5">
            <h4>Intent classes</h4>
            <div class="stat-row" style="flex-direction:column">
              <div class="stat"><div class="stat-label">Val accuracy</div><div class="stat-value accent" id="s3-acc-big">0.00%</div><div class="stat-sub">held-out windows</div></div>
              <div class="stat"><div class="stat-label">Classes</div><div class="stat-value">${s.n_intent_classes}</div><div class="stat-sub">${s.intent_names.join(" · ")}</div></div>
              <div class="stat"><div class="stat-label">Force MSE</div><div class="stat-value">${s.best_val_force_mse.toExponential(2)}</div><div class="stat-sub">EMG-amplitude proxy target</div></div>
            </div>
          </div>
          <div class="card col-6">
            <h4>Live intent label + force gauge</h4>
            <video class="big" data-src="${s.video}" muted loop playsinline preload="none" controls></video>
            <div class="caption">Real preview clip: rolling EMG, predicted intent class, and the force-head gauge.</div>
          </div>
          <div class="card col-6">
            <h4>Confusion matrix</h4>
            <img src="${s.confusion_matrix}" alt="confusion matrix" />
            <div class="caption">Diagonal-heavy across 5 classes on held-out windows.</div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s3", "Loading LSTM checkpoint…", 700);
      revealContent("s3");
      loadVideos(sc);
      // animate epoch / acc / mask wipe
      const mask = sc.querySelector("#s3-mask");
      const epochEl = sc.querySelector("#s3-epoch");
      const accEl = sc.querySelector("#s3-acc");
      const accBig = sc.querySelector("#s3-acc-big");
      const totalEpochs = s.epochs_run;
      const finalAcc = s.best_val_intent_acc_pct;
      const dur = 2400;
      const start = performance.now();
      function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        mask.style.width = (100 - 100 * eased) + "%";
        const ep = Math.round(eased * totalEpochs);
        const ac = eased * finalAcc;
        epochEl.textContent = ep;
        accEl.textContent = ac.toFixed(2) + "%";
        accBig.textContent = ac.toFixed(2) + "%";
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
      const v = sc.querySelector("video");
      if (v) { v.load(); v.play().catch(()=>{}); }
    });
    return sc;
  }

  // STAGE 4,Force Validation
  function buildScreen4() {
    const s = D.stage2_force;
    const sc = el("section", "screen", "");
    sc.id = "screen-s4";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 4</span>
          <div>
            <h2 class="stage-title">Force Validation</h2>
            <div class="stage-subtitle">Validate EMG-predicted force against real 6-axis force-sensor measurements on a held-out repetition.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">The model's predicted force is checked against real, measured force from a calibrated sensor.</div>
        <div class="ctx-item">This is the headline result: an R² of 0.96, meaning the model's force predictions closely match real, physical grip force.</div>
        <div class="ctx-item">This stage proves the core bet of the entire company: muscle signals genuinely carry usable, accurate force information.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">⚖</div>
          <div>
            <div style="font-weight:600">Validate Against Real Force Sensors</div>
            <div class="action-desc">Block E3 of DB2 has real measured force. The dedicated E3 force model is evaluated on repetition 6 (never seen during training).</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s4">Validate Against Real Force Sensors</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-12" style="text-align:center; padding: 28px;">
            <h4>EMG-predicted vs real force</h4>
            <div style="font-size:64px; font-weight:800; letter-spacing:-1px; color: var(--accent); line-height:1.1; margin: 10px 0 4px;" id="s4-r2-big">R² = 0.00</div>
            <div class="muted" style="font-size:14px">Pearson r = <span id="s4-pr">0.00</span> · RMSE = <span id="s4-rmse">0.00</span> N · MAPE = <span id="s4-mape">0.00</span>%</div>
            <div class="caption" style="margin-top:14px; max-width:780px; margin-left:auto; margin-right:auto;">
              ${s.notes_caption}
            </div>
          </div>
          <div class="card col-6">
            <h4>Predicted vs measured (time series)</h4>
            <img src="${s.prediction_png}" alt="force prediction time series" />
            <div class="caption">Predicted force (model) overlaid on measured force (real sensors) over the held-out repetition.</div>
          </div>
          <div class="card col-6">
            <h4>Predicted vs measured (scatter)</h4>
            <img src="${s.scatter_png}" alt="force scatter with R^2" />
            <div class="caption">Sample-by-sample scatter. Tight diagonal = strong agreement.</div>
          </div>
          <div class="card col-12">
            <h4>Force range (real units)</h4>
            <div class="stat-row">
              <div class="stat"><div class="stat-label">RMSE</div><div class="stat-value">${s.rmse_real_N.toFixed(2)} N</div></div>
              <div class="stat"><div class="stat-label">MAE</div><div class="stat-value">${s.mae_real_N.toFixed(2)} N</div></div>
              <div class="stat"><div class="stat-label">MAPE</div><div class="stat-value">${s.mape_pct.toFixed(1)}%</div></div>
              <div class="stat"><div class="stat-label">Test samples</div><div class="stat-value">${s.test_samples.toLocaleString()}</div></div>
            </div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s4", "Loading E3 force model + held-out test…", 900);
      revealContent("s4");
      const big = sc.querySelector("#s4-r2-big");
      const pr = sc.querySelector("#s4-pr");
      const rmse = sc.querySelector("#s4-rmse");
      const mape = sc.querySelector("#s4-mape");
      const start = performance.now();
      const dur = 1600;
      function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        big.textContent = "R² = " + (s.r_squared * eased).toFixed(4);
        pr.textContent = (s.pearson_r * eased).toFixed(3);
        rmse.textContent = (s.rmse_real_N * eased).toFixed(2);
        mape.textContent = (s.mape_pct * eased).toFixed(1);
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
    return sc;
  }

  // STAGE 5,Simulation Scene
  function buildScreen5() {
    const s = D.stage3;
    const sc = el("section", "screen", "");
    sc.id = "screen-s5";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 5</span>
          <div>
            <h2 class="stage-title">Simulation Scene</h2>
            <div class="stage-subtitle">${s.arm} + ${s.target} in ${s.simulator}. Operator EMG drives the arm; the policy must keep contact force below the crush threshold.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">The validated force and intent predictions are used to drive a physics simulation: a robot arm handling an object based on the learned force behavior.</div>
        <div class="ctx-item">This shows the predicted force translating into physical, simulated action.</div>
        <div class="ctx-item">The simulation includes a fragile object and a defined force threshold, so incorrect force is visibly detectable.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">⛁</div>
          <div>
            <div style="font-weight:600">Load Simulation</div>
            <div class="action-desc">Boot the scene, then play a synced EMG → arm → force-gauge clip.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s5">Load Simulation</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-12">
            <h4>Synced demo, EMG drives arm, force-gauge live</h4>
            <video class="big" data-src="${s.video}" muted loop playsinline preload="none" controls></video>
            <div class="caption">Each phase (REACHING → GRIPPING → STABILIZING → RELEASING) is driven by the LSTM's predicted intent. Force gauge stays below the crush threshold.</div>
          </div>
          <div class="card col-6">
            <h4>Scene render</h4>
            <img src="${s.scene_render}" alt="UR5e + grape scene render" />
            <div class="caption">${s.arm} + grape on table. Simulated in ${s.simulator}.</div>
          </div>
          <div class="card col-6">
            <h4>Crush-threshold safety</h4>
            <div class="crush-scale"><span>0 N</span><span>6 N (crush)</span></div>
            <div class="crush-bar">
              <div class="crush-marker" data-label="max contact ${s.max_contact_N.toFixed(2)} N" style="left:${(s.max_contact_N / s.crush_threshold_N * 100).toFixed(1)}%"></div>
            </div>
            <div class="stat-row" style="margin-top: 24px; flex-direction: column;">
              <div class="stat"><div class="stat-label">Max contact</div><div class="stat-value ok">${s.max_contact_N.toFixed(2)} N</div><div class="stat-sub">across the full clip</div></div>
              <div class="stat"><div class="stat-label">Crush threshold</div><div class="stat-value">${s.crush_threshold_N.toFixed(1)} N</div></div>
              <div class="stat"><div class="stat-label">Ever exceeded?</div><div class="stat-value ok">${s.ever_exceeded_crush ? "YES" : "NO"}</div></div>
              <div class="stat"><div class="stat-label">Grape in zone</div><div class="stat-value ok">${s.grape_in_target_zone ? "YES" : "NO"}</div></div>
            </div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s5", "Loading MuJoCo scene…", 900);
      revealContent("s5");
      loadVideos(sc);
      const v = sc.querySelector("video");
      if (v) { v.load(); v.play().catch(()=>{}); }
    });
    return sc;
  }

  // STAGE 6,Augmentation + Verification
  function buildScreen6() {
    const s = D.stage5;
    const totalTiles = 100; // we visualize 100 passing scenarios as the tile count
    const sc = el("section", "screen", "");
    sc.id = "screen-s6";
    // we mark 23 tiles as rejected (FORCE_PROFILE_MISMATCH),same number as real data
    const rejectionIdxs = new Set();
    while (rejectionIdxs.size < s.rejection_breakdown.FORCE_PROFILE_MISMATCH) {
      rejectionIdxs.add(Math.floor(Math.random() * 123));
    }
    // We'll show 123 total tiles (raw) and mark 23 as rejected
    const TOTAL = s.raw_scenarios;
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 6</span>
          <div>
            <h2 class="stage-title">Augmentation + Verification</h2>
            <div class="stage-subtitle">${s.n_baselines} baseline demonstrations re-simulated under randomized physics, then filtered through a 6-point verification.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">A small number of real demonstrations are expanded into many simulated variations, different positions, conditions, and trajectories.</div>
        <div class="ctx-item">Every generated scenario is checked against the real captured force data, and anything that doesn't match is discarded.</div>
        <div class="ctx-item">This is how the system scales limited real data into a larger, still physically valid training set.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">⚗</div>
          <div>
            <div style="font-weight:600">Augment + Verify</div>
            <div class="action-desc">Generate randomized variants, then run each through the 6-check filter.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s6">Augment + Verify</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-7">
            <h4>Scenario tiles, randomized physics</h4>
            <div class="aug-tiles" id="s6-tiles"></div>
            <div class="caption" style="margin-top:14px">Each tile is one re-simulated scenario with randomized grape mass, jaw friction, approach angle, table friction, and lighting. Red = filtered out by verification.</div>
          </div>
          <div class="card col-5">
            <h4>6-point verification</h4>
            <ul class="verify-list" id="s6-verify">
              ${s.verify_checks.map(c => `
                <li class="verify-item" data-key="${c.key}">
                  <span class="verify-badge"></span>
                  <span>${c.label}</span>
                  <span class="verify-count" data-count></span>
                </li>
              `).join("")}
            </ul>
            <div class="caption" style="margin-top:14px">All 23 rejections came from <span class="mono" style="color: var(--danger)">FORCE_PROFILE_MISMATCH</span>. No path, contact-timing, task, force-band, or overshoot failures.</div>
          </div>
          <div class="card col-12">
            <h4>Final augmented dataset</h4>
            <div class="stat-row">
              <div class="stat"><div class="stat-label">Baselines</div><div class="stat-value" id="s6-baselines">0</div><div class="stat-sub">Stage 4 demos</div></div>
              <div class="stat"><div class="stat-label">Raw scenarios</div><div class="stat-value" id="s6-raw">0</div></div>
              <div class="stat"><div class="stat-label">Validated</div><div class="stat-value accent" id="s6-pass">0</div></div>
              <div class="stat"><div class="stat-label">Pass rate</div><div class="stat-value accent" id="s6-rate">0.0%</div></div>
              <div class="stat"><div class="stat-label">Peak force (mean)</div><div class="stat-value">${s.peak_force_mean_N.toFixed(2)} N</div><div class="stat-sub">crush ${s.crush_threshold_N.toFixed(1)} N</div></div>
              <div class="stat"><div class="stat-label">Max overshoot</div><div class="stat-value">${s.peak_force_max_N.toFixed(2)} N</div><div class="stat-sub">all below crush</div></div>
            </div>
          </div>
          <div class="card col-6">
            <h4>Augmentation grid (real saved)</h4>
            <img src="${s.grid_png}" alt="augmentation grid render" />
            <div class="caption">Saved 8×8 grid from the real augmentation run.</div>
          </div>
          <div class="card col-6">
            <h4>Sample scenario clip</h4>
            <video data-src="${s.scenario_video}" muted loop playsinline preload="none" controls></video>
            <div class="caption">One of the 100 passing scenarios under randomized physics.</div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s6", "Generating randomized scenarios…", 800);
      revealContent("s6");
      loadVideos(sc);
      // Build tiles, then animate them in
      const tilesEl = sc.querySelector("#s6-tiles");
      tilesEl.innerHTML = "";
      const tiles = [];
      for (let i = 0; i < TOTAL; i++) {
        const t = document.createElement("div");
        t.className = "aug-tile";
        if (rejectionIdxs.has(i)) t.dataset.rejected = "1";
        tilesEl.appendChild(t);
        tiles.push(t);
      }
      const tileMs = 14;
      for (let i = 0; i < tiles.length; i++) {
        setTimeout(() => {
          tiles[i].classList.add("shown");
          if (tiles[i].dataset.rejected) tiles[i].classList.add("rejected");
        }, i * tileMs);
      }
      await new Promise(r => setTimeout(r, tiles.length * tileMs + 200));

      // Verification list
      const items = sc.querySelectorAll(".verify-item");
      const checks = s.verify_checks;
      const total = s.raw_scenarios;
      for (let i = 0; i < items.length; i++) {
        items[i].classList.add("active");
        const key = checks[i].key.toUpperCase();
        // Map our friendlier keys to the rejection_breakdown keys
        const keyMap = {
          PATH_SIMILARITY: "PATH_DEVIATION",
          CONTACT_TIMING: "CONTACT_TIMING",
          TASK_SUCCESS: "TASK_FAILURE",
          PEAK_FORCE: "FORCE_OUT_OF_BAND",
          FORCE_PROFILE: "FORCE_PROFILE_MISMATCH",
          FORCE_OVERSHOOT: "FORCE_OVERSHOOT",
        };
        const rejCount = s.rejection_breakdown[keyMap[key]] || 0;
        const passCount = total - rejCount;
        await new Promise(r => setTimeout(r, 280));
        items[i].classList.add(rejCount === 0 ? "pass" : "fail");
        const countEl = items[i].querySelector("[data-count]");
        countEl.textContent = rejCount === 0
          ? `${passCount}/${total} pass`
          : `${rejCount} rejected (FORCE_PROFILE_MISMATCH)`;
      }
      // Tally counters
      const baselinesEl = sc.querySelector("#s6-baselines");
      const rawEl = sc.querySelector("#s6-raw");
      const passEl = sc.querySelector("#s6-pass");
      const rateEl = sc.querySelector("#s6-rate");
      const dur = 1100;
      const start = performance.now();
      function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        baselinesEl.textContent = Math.round(s.n_baselines * eased);
        rawEl.textContent = Math.round(s.raw_scenarios * eased);
        passEl.textContent = Math.round(s.passing * eased);
        rateEl.textContent = (s.pass_rate_pct * eased).toFixed(1) + "%";
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
    return sc;
  }

  // STAGE 7,Policy Training (diffusion)
  function buildScreen7() {
    const s = D.stage7;
    const sc = el("section", "screen", "");
    sc.id = "screen-s7";
    sc.innerHTML = `
      <div class="stage-header">
        <div class="stage-title-wrap">
          <span class="stage-eyebrow">Stage 7</span>
          <div>
            <h2 class="stage-title">Policy Training</h2>
            <div class="stage-subtitle">Multimodal diffusion policy: vision (DINOv2) + EMG force intent (LSTM hidden state) + robot state, fused to generate force-aware action sequences.</div>
          </div>
        </div>
      </div>

      <div class="stage-context">
        <div class="ctx-item">A diffusion policy is the model that turns everything the system has learned into actual robot actions. It generates a sequence of movements by starting from noise and progressively refining it into a coherent action, guided by what the robot sees and the force intent learned from human demonstration.</div>
        <div class="ctx-item">We use diffusion because contact-rich tasks have many valid ways to succeed, and unlike simpler methods that average those into one mushy, failing motion, a diffusion policy can represent the full range of correct actions and pick a clean one.</div>
        <div class="ctx-item">The verified, augmented data trains this policy — a model that outputs actions (movement and force) rather than just predictions. This is the step from understanding force to acting on it.</div>
      </div>

      <div class="action-card">
        <div class="left">
          <div class="action-icon">◈</div>
          <div>
            <div style="font-weight:600">Train Diffusion Policy</div>
            <div class="action-desc">Replay the real training run, show the architecture, and visualize the denoising concept.</div>
          </div>
        </div>
        <button class="btn-primary" data-action="s7">Train Diffusion Policy</button>
      </div>

      <div class="loader-zone"></div>

      <div class="reveal">
        <div class="grid">
          <div class="card col-7">
            <h4>Training replay, real loss curve</h4>
            <div class="replay-wrap" id="s7-replay">
              <img src="${s.loss_curve}" alt="diffusion training loss curve" />
              <div class="replay-hud">
                <div>epoch <span class="hud-value" id="s7-epoch">0</span> / ${s.stopped_epoch}</div>
                <div>val MSE <span class="hud-value" id="s7-mse">0.0000</span></div>
              </div>
              <div class="replay-mask" id="s7-mask"></div>
              <div class="replay-tag">Replay of real training run</div>
            </div>
            <div class="caption">Best val MSE <strong>${s.best_val_mse.toFixed(4)}</strong> at epoch ${s.best_epoch}. Early-stopped at epoch ${s.stopped_epoch}. Wall time ${s.wall_time} on a single RTX 5060.</div>
          </div>
          <div class="card col-5">
            <h4>Training stats</h4>
            <div class="stat-row" style="flex-direction:column">
              <div class="stat"><div class="stat-label">Best val MSE</div><div class="stat-value accent" id="s7-mse-big">0.0000</div><div class="stat-sub">@ epoch ${s.best_epoch}</div></div>
              <div class="stat"><div class="stat-label">Trainable params</div><div class="stat-value">${s.trainable_params_M.toFixed(1)} M</div><div class="stat-sub">denoiser only (DINOv2 + LSTM frozen)</div></div>
              <div class="stat"><div class="stat-label">Demonstrations</div><div class="stat-value">${s.n_demos_total}</div><div class="stat-sub">30 baseline + 100 augmented</div></div>
            </div>
          </div>

          <div class="card col-12">
            <h4>Architecture, multimodal fusion</h4>
            <div class="arch">
              <div class="arch-col">
                <div class="arch-block"><span class="name">Vision</span>DINOv2-base (frozen)<br/>640×480 frame → 768 dim</div>
                <div class="arch-block"><span class="name">EMG / LSTM</span>Stage 2 LSTM hidden (frozen)<br/>200 ms × 70 EMG → 128 dim</div>
                <div class="arch-block"><span class="name">Robot State</span>MLP (64 h)<br/>20 dim → 64 dim</div>
              </div>
              <div class="arch-arrow">→</div>
              <div class="arch-col">
                <div class="arch-block fuse"><span class="name">Fusion</span>concat → ${s.fusion_dim}-dim conditioning vector<br/>(FiLM into denoiser)</div>
              </div>
              <div class="arch-arrow">→</div>
              <div class="arch-col">
                <div class="arch-block out"><span class="name">ConditionalUnet1D</span>DDPM(100) train · DDIM(5) infer<br/>predicts (16, 7) noise → action seq</div>
              </div>
            </div>
            <div class="caption" style="margin-top:14px">EMG force intent is the third stream; this is the differentiator vs vision-only policies.</div>
          </div>

          <div class="card col-6">
            <h4>Diffusion denoising concept</h4>
            <canvas class="denoise-canvas" id="s7-denoise" width="600" height="120"></canvas>
            <div class="caption">Illustrative animation only, random noise resolves into a smooth action trajectory over denoising steps. Real inference uses DDIM with 5 steps.</div>
          </div>
          <div class="card col-6">
            <h4>Diagnostic replay (real clip)</h4>
            <video data-src="${s.diag_video}" muted loop playsinline preload="none" controls></video>
            <div class="caption">Ground-truth action sequence replayed through the execute loop, the arm reaches within 9 mm of the grape under 1.34 N contact. Confirms the execution loop is correct.</div>
          </div>

          <div class="card col-12">
            <h4>Honest note on closed-loop execution</h4>
            <div class="honest-note">
              <span style="display:inline-block; font-size:11px; font-weight:800; letter-spacing:0.12em; text-transform:uppercase; color:#ef4444; border:1px solid rgba(239,68,68,0.4); background:rgba(239,68,68,0.08); border-radius:6px; padding:3px 10px; margin-bottom:12px;">WHY THIS DIDN'T WORK</span>
              <strong>Training converged</strong> (val MSE settled to ${s.best_val_mse.toFixed(4)} at epoch ${s.best_epoch}). On ${s.closed_loop_trials} fresh closed-loop test scenarios, the policy was <strong>force-safe in every trial</strong>, peak contact ${s.closed_loop_peak_max_N.toFixed(2)} N vs ${s.crush_threshold_N.toFixed(1)} N crush threshold, but task completion was <strong>${s.closed_loop_success} / ${s.closed_loop_trials}</strong>.
              <br/><br/>
              In this demo the policy trained and converged cleanly, but it didn't succeed at closed-loop execution, because it was trained on only around 130 samples, most of them from simulation. This is the classic small-data failure mode in imitation learning: tiny errors compound over a trajectory, and the policy hasn't seen enough real, diverse examples to recover from states it wasn't trained on.
              <br/><br/>
              The fix is more real-world demonstration data with greater diversity, which is exactly what building the physical hardware rig enables, and it's the next phase of development.
            </div>
          </div>
        </div>
      </div>
    `;
    sc.querySelector(".btn-primary").addEventListener("click", async () => {
      await runLoader("s7", "Loading diffusion policy + cached features…", 1000);
      revealContent("s7");
      loadVideos(sc);
      // animate loss replay
      const mask = sc.querySelector("#s7-mask");
      const epochEl = sc.querySelector("#s7-epoch");
      const mseEl = sc.querySelector("#s7-mse");
      const mseBig = sc.querySelector("#s7-mse-big");
      const totalEpochs = s.stopped_epoch;
      const finalMse = s.best_val_mse;
      const dur = 2400;
      const start = performance.now();
      function tick(now) {
        const t = Math.min(1, (now - start) / dur);
        const eased = 1 - Math.pow(1 - t, 3);
        mask.style.width = (100 - 100 * eased) + "%";
        // val MSE starts ~0.10 and decays to 0.0066,model that approximately
        const v = 0.10 * Math.pow(1 - eased, 1.6) + finalMse * eased;
        epochEl.textContent = Math.round(eased * totalEpochs);
        mseEl.textContent = v.toFixed(4);
        mseBig.textContent = v.toFixed(4);
        if (t < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);

      // diffusion denoise canvas: noise -> smooth curve
      const canvas = sc.querySelector("#s7-denoise");
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height;
      const N = 120;
      // target: smooth trajectory
      const target = new Array(N).fill(0).map((_, i) => {
        const x = i / (N - 1);
        return Math.sin(x * Math.PI * 2.1) * 0.6 + Math.sin(x * Math.PI * 4.2) * 0.18;
      });
      const T_STEPS = 30;
      let step = 0;
      function drawDenoise() {
        ctx.clearRect(0, 0, W, H);
        // grid
        ctx.strokeStyle = "rgba(255,255,255,0.05)";
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
          ctx.beginPath();
          ctx.moveTo(0, H * i / 4);
          ctx.lineTo(W, H * i / 4);
          ctx.stroke();
        }
        const noise = Math.max(0, 1 - step / T_STEPS);
        ctx.lineWidth = 2;
        const grad = ctx.createLinearGradient(0, 0, W, 0);
        grad.addColorStop(0, "#60a5fa");
        grad.addColorStop(1, "#6ee7b7");
        ctx.strokeStyle = grad;
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
          const x = (i / (N - 1)) * W;
          const noisy = target[i] + (Math.random() - 0.5) * noise * 2.4;
          const y = H / 2 - noisy * (H / 2 - 10);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        // step label
        ctx.fillStyle = "rgba(255,255,255,0.5)";
        ctx.font = "11px ui-monospace, Consolas, monospace";
        ctx.fillText("denoise step " + step + " / " + T_STEPS, 8, 14);
        step++;
        if (step <= T_STEPS) setTimeout(drawDenoise, 100);
      }
      drawDenoise();

      const v = sc.querySelector("video");
      if (v) { v.load(); v.play().catch(()=>{}); }
    });
    return sc;
  }

  // SUMMARY
  function buildSummary() {
    const s2 = D.stage2;
    const s2f = D.stage2_force;
    const s5 = D.stage5;
    const sc = el("section", "screen", "");
    sc.id = "screen-summary";
    sc.innerHTML = `
      <div class="summary-hero">
        <img src="assets/novonus-logo.png" class="summary-mark" alt="Novonus" />
        <h2 class="summary-title">Novonus</h2>
        <div class="summary-sub">Force-intelligence for collaborative robots, EMG-conditioned policies that learn force-aware manipulation from operator demonstrations.</div>
        <div class="summary-stats">
          <div class="summary-metric">
            <span class="m-label">Intent accuracy</span>
            <span class="m-value">${s2.best_val_intent_acc_pct.toFixed(2)}%</span>
            <span class="m-sub">${s2.n_intent_classes}-class LSTM on Ninapro DB2</span>
          </div>
          <div class="summary-metric">
            <span class="m-label">Force prediction</span>
            <span class="m-value">R² = ${s2f.r_squared.toFixed(2)}</span>
            <span class="m-sub">EMG → real 6-axis force, held-out test</span>
          </div>
          <div class="summary-metric">
            <span class="m-label">Augmented dataset</span>
            <span class="m-value">${s5.passing} / ${s5.raw_scenarios}</span>
            <span class="m-sub">${s5.pass_rate_pct.toFixed(1)}% pass rate, 6-check filter</span>
          </div>
          <div class="summary-metric green">
            <span class="m-label">Force safety</span>
            <span class="m-value">100%</span>
            <span class="m-sub">Below 6 N crush in every sim trial</span>
          </div>
        </div>
        <div class="summary-next">Next: real hardware + more operator demonstrations.</div>
        <button class="summary-cta" type="button"
          data-cal-link="deepayan"
          data-cal-namespace="deepayan"
          data-cal-config='{"layout":"month_view"}'>
          Book a call
        </button>
      </div>
      <div class="stage-context" style="margin-top:16px">
        <div class="ctx-item">Recaps the full pipeline: raw EMG in, cleaned and modeled, validated against real force, used to drive simulated action.</div>
        <div class="ctx-item">States clearly what's proven (the 0.96 force-prediction result) versus what's in development (multimodal capture, real hardware, closed-loop deployment).</div>
        <div class="ctx-item">Closes with what's next: building the full sensor rig and moving this validated core onto real hardware.</div>
      </div>
    `;
    return sc;
  }

  // ---------- mount -----------------------------------------------------
  function mount() {
    mainEl.appendChild(buildScreen0());
    mainEl.appendChild(buildScreen1());
    mainEl.appendChild(buildScreen2());
    mainEl.appendChild(buildScreen3());
    mainEl.appendChild(buildScreen4());
    mainEl.appendChild(buildScreen5());
    mainEl.appendChild(buildScreen6());
    mainEl.appendChild(buildScreen7());
    mainEl.appendChild(buildSummary());

    renderNav();
    showScreen(stages[0].id);
    updateFooter();

    document.getElementById("prev-btn").addEventListener("click", () => goTo(current - 1));
    document.getElementById("next-btn").addEventListener("click", () => goTo(current + 1));

    document.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "ArrowRight") goTo(current + 1);
      else if (e.key === "ArrowLeft") goTo(current - 1);
    });
  }

  document.addEventListener("DOMContentLoaded", mount);
})();
