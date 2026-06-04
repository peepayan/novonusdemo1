"""Verify Novonus dev environment: CUDA + MuJoCo Warp + DINOv2."""

from __future__ import annotations

import sys
import traceback

results: dict[str, tuple[bool, str]] = {}


def check(name: str):
    def deco(fn):
        print(f"\n=== {name} ===")
        try:
            msg = fn() or "ok"
            results[name] = (True, msg)
            print(f"[PASS] {name}: {msg}")
        except Exception as e:
            results[name] = (False, f"{type(e).__name__}: {e}")
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
        return fn
    return deco


@check("torch + CUDA")
def _torch():
    import torch
    info = [f"torch={torch.__version__}", f"cuda_runtime={torch.version.cuda}"]
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    props = torch.cuda.get_device_properties(0)
    info.append(f"gpu='{torch.cuda.get_device_name(0)}'")
    info.append(f"capability=sm_{props.major}{props.minor}")
    info.append(f"vram={props.total_memory / 1024**3:.2f} GB")
    # Real kernel launch on Blackwell to catch sm_120 mismatch.
    x = torch.randn(128, 128, device="cuda")
    _ = (x @ x).sum().item()
    torch.cuda.synchronize()
    info.append("matmul=ok")
    return " | ".join(info)


@check("MuJoCo Warp (GPU physics)")
def _mjw():
    import warp as wp
    import mujoco
    import mujoco_warp

    wp.init()
    if not any(str(d).startswith("cuda") for d in wp.get_devices()):
        raise RuntimeError("warp sees no CUDA device")

    xml = """
    <mujoco>
      <worldbody>
        <light pos='0 0 1'/>
        <geom type='plane' size='1 1 0.05'/>
        <body pos='0 0 0.3'>
          <joint type='free'/>
          <geom type='sphere' size='0.05'/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm = mujoco.MjModel.from_xml_string(xml)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)

    # Step on GPU via mujoco_warp.
    with wp.ScopedDevice("cuda:0"):
        m = mujoco_warp.put_model(mjm)
        d = mujoco_warp.put_data(mjm, mjd)
        mujoco_warp.step(m, d)
        wp.synchronize()

    # Render one offscreen frame (confirms scene pipeline).
    with mujoco.Renderer(mjm, width=64, height=64) as r:
        r.update_scene(mjd)
        frame = r.render()
    return f"gpu_step=ok, frame.shape={tuple(frame.shape)}"


@check("DINOv2 (HF transformers)")
def _dino():
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov2-base"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Dummy image: 224x224 RGB noise.
    import numpy as np
    arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
    img = Image.fromarray(arr)
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    feats = out.last_hidden_state
    return f"model={model_id} | device={device} | last_hidden_state.shape={tuple(feats.shape)}"


def main() -> int:
    print("\n" + "=" * 60)
    failed = [name for name, (ok, _) in results.items() if not ok]
    if failed:
        print("FAILED CHECKS:")
        for name in failed:
            print(f"  - {name}: {results[name][1]}")
        print("=" * 60)
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
