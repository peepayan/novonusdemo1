"""Build the Stage 3 MuJoCo scene XML.

Programmatically extends the Menagerie UR5e model with:
    - a table (~0.8 x 0.6 m, height 0.75 m)
    - a fragile grape (low mass, soft contact, 6 N crush threshold)
    - a target zone disc on the table
    - a fixed 45-degree camera (640 x 480) named ``scene_cam``
    - a simple 2-jaw parallel gripper attached to the wrist_3 attachment_site
      driven by position actuators so we can read grip force from the LSTM and
      command it directly

The XML is written to ``data/assets/scene.xml`` once, and every downstream
stage loads from that single canonical file.
"""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from . import physics_constants as pc


# ---------------------------------------------------------------------------
# XML builder helpers
# ---------------------------------------------------------------------------

def _set(elem: ET.Element, **attrs) -> ET.Element:
    for k, v in attrs.items():
        if isinstance(v, (tuple, list)):
            elem.set(k, " ".join(f"{x:.6g}" for x in v))
        elif isinstance(v, float):
            elem.set(k, f"{v:.6g}")
        else:
            elem.set(k, str(v))
    return elem


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return _set(ET.SubElement(parent, tag), **attrs)


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(f"body {name!r} not found")


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene_xml(ur5e_xml: Path = pc.UR5E_XML,
                    out_path: Path = pc.SCENE_XML_OUT) -> Path:
    """Build ``scene.xml`` next to the ``ur5e/`` mesh directory and return the
    path. The compiler's ``meshdir`` is rewritten so the mesh paths still
    resolve when the file lives at ``data/assets/scene.xml``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(ur5e_xml)
    root = tree.getroot()
    root.set("model", "novonus_stage3_scene")

    # rewrite meshdir to be relative to scene.xml's location
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", "ur5e/assets")
    comp.set("angle", "radian")
    comp.set("autolimits", "true")

    # global options: gravity + 500 Hz timestep + GPU-friendly integrator
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    _set(opt,
         timestep=pc.SIM_TIMESTEP_S,
         gravity=(0.0, 0.0, -9.81),
         integrator="implicitfast",
         iterations=20,
         tolerance=1e-8)

    # visual defaults — dark headlight so the scene reads cleanly on screen
    visual = root.find("visual") or ET.SubElement(root, "visual")
    _sub(visual, "headlight",
         ambient=(0.35, 0.35, 0.35),
         diffuse=(0.5, 0.5, 0.5),
         specular=(0.2, 0.2, 0.2))
    _sub(visual, "rgba", haze=(0.15, 0.20, 0.25, 1.0))
    _sub(visual, "global", offwidth=pc.RENDER_W, offheight=pc.RENDER_H)

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # sky + floor + table textures/materials
    _sub(asset, "texture", type="skybox", builtin="gradient",
         rgb1=(0.08, 0.09, 0.12), rgb2=(0.02, 0.02, 0.04),
         width=256, height=256)
    _sub(asset, "texture", name="floor_tex", type="2d", builtin="checker",
         rgb1=(0.10, 0.10, 0.10), rgb2=(0.16, 0.16, 0.16),
         width=512, height=512, mark="cross",
         markrgb=(0.30, 0.30, 0.30))
    _sub(asset, "material", name="floor_mat", texture="floor_tex",
         texrepeat=(4, 4), reflectance=0.05)
    _sub(asset, "material", name="table_mat",
         rgba=(0.55, 0.42, 0.30, 1.0), specular=0.2, shininess=0.4)
    _sub(asset, "material", name="grape_mat",
         rgba=(0.42, 0.10, 0.45, 1.0), specular=0.6, shininess=0.7,
         reflectance=0.05)
    _sub(asset, "material", name="target_mat",
         rgba=(0.20, 0.85, 0.45, 0.6), specular=0.1)
    _sub(asset, "material", name="jaw_mat",
         rgba=(0.10, 0.10, 0.10, 1.0), specular=0.6, shininess=0.4)

    wb = root.find("worldbody")
    assert wb is not None, "ur5e.xml must define a worldbody"

    # ground light + floor
    _sub(wb, "light", name="key_light", pos=(0.4, -0.6, 2.0),
         dir=(-0.2, 0.4, -1.0), diffuse=(0.7, 0.7, 0.7))
    _sub(wb, "geom", name="floor", type="plane",
         size=(2.0, 2.0, 0.05), material="floor_mat",
         contype=1, conaffinity=1, group=1)

    # table — a static box sitting on the floor
    table_hx, table_hy, table_hz = pc.TABLE_SIZE_XYZ
    table_z = pc.TABLE_HEIGHT - table_hz / 2.0
    cx, cy = pc.TABLE_CENTER_XY
    _sub(wb, "body", name="table", pos=(cx, cy, table_z))
    table = wb.find("body[@name='table']")
    _sub(table, "geom", name="table_top", type="box",
         size=(table_hx, table_hy, table_hz / 2.0),
         material="table_mat", contype=1, conaffinity=1,
         friction=(0.8, 0.05, 0.001), group=1)
    # the table top surface z (used to place the grape and target zone)
    surface_z = table_z + table_hz / 2.0

    # arm base mount — a small pedestal so the arm's base does not clip the
    # floor. The UR5e's own root body is named "base"; we add a static
    # pedestal under it.
    arm_x, arm_y = pc.ARM_BASE_XY
    _sub(wb, "body", name="arm_pedestal",
         pos=(arm_x, arm_y, pc.ARM_BASE_Z / 2.0))
    pedestal = wb.find("body[@name='arm_pedestal']")
    _sub(pedestal, "geom", name="pedestal_geom", type="cylinder",
         size=(0.08, pc.ARM_BASE_Z / 2.0),
         material="jaw_mat", contype=1, conaffinity=1, group=1)

    # raise the UR5e root body so its base sits on top of the pedestal.
    base = _find_body(root, "base")
    base.set("pos", f"{arm_x:.6g} {arm_y:.6g} {pc.ARM_BASE_Z:.6g}")
    base.set("quat", "1 0 0 0")  # face +x toward the table

    # target zone — a thin disc on the table surface
    tx, ty = pc.TARGET_ZONE_XY
    _sub(wb, "body", name="target_zone", pos=(tx, ty, surface_z + 0.0005))
    target = wb.find("body[@name='target_zone']")
    _sub(target, "geom", name="target_geom", type="cylinder",
         size=(pc.TARGET_ZONE_RADIUS, 0.0005),
         material="target_mat",
         contype=0, conaffinity=0, group=1)

    # grape — small free body with low mass, soft contact, friction
    gx, gy = pc.GRAPE_INIT_XY
    gz = surface_z + pc.GRAPE_RADIUS + 0.001
    _sub(wb, "body", name="grape", pos=(gx, gy, gz))
    grape = wb.find("body[@name='grape']")
    _sub(grape, "freejoint", name="grape_free")
    _sub(grape, "inertial", pos=(0, 0, 0), mass=pc.GRAPE_MASS_KG,
         diaginertia=(2e-6, 2e-6, 2e-6))
    _sub(grape, "geom", name="grape_geom", type="sphere",
         size=(pc.GRAPE_RADIUS,),
         material="grape_mat",
         friction=pc.GRAPE_FRICTION,
         solref=pc.GRAPE_SOLREF,
         solimp=pc.GRAPE_SOLIMP,
         contype=1, conaffinity=1, group=1)

    # camera — 45° from above and in front of the scene, framing the arm,
    # the table, the grape, and the target zone. We use a body-targeted
    # camera on a small "cam_target" body sitting just above the table
    # center so framing is robust to the grape getting picked up.
    _sub(wb, "body", name="cam_target", pos=(cx - 0.05, cy, surface_z + 0.15))
    cam_target = wb.find("body[@name='cam_target']")
    _sub(cam_target, "geom", name="cam_target_geom", type="sphere",
         size=(0.0001,), rgba=(0, 0, 0, 0), contype=0, conaffinity=0, group=4)

    cam_pos = (-0.30, -0.95, 1.55)
    _sub(wb, "camera", name=pc.CAMERA_NAME,
         pos=cam_pos, mode="targetbody", target="cam_target",
         fovy=pc.CAMERA_FOVY_DEG)

    # gripper jaws — attached to wrist_3_link as children. Two boxes that
    # slide inward via prismatic joints. Position actuators are added below.
    wrist3 = _find_body(root, "wrist_3_link")
    # left jaw (slides along +x in local frame)
    _sub(wrist3, "body", name="left_jaw",
         pos=(0.030, 0.110, 0.0))
    left_jaw = wrist3.find("body[@name='left_jaw']")
    _sub(left_jaw, "joint", name="left_jaw_slide", type="slide",
         axis=(-1, 0, 0), range=(0.0, 0.045),
         damping=8.0, stiffness=0.0, armature=0.001)
    _sub(left_jaw, "inertial", pos=(0, 0, 0), mass=0.02,
         diaginertia=(1e-5, 1e-5, 1e-5))
    _sub(left_jaw, "geom", name="left_jaw_geom", type="box",
         size=(0.005, 0.012, 0.025),
         material="jaw_mat",
         friction=(1.0, 0.05, 0.001),
         contype=1, conaffinity=1, group=1)

    _sub(wrist3, "body", name="right_jaw",
         pos=(-0.030, 0.110, 0.0))
    right_jaw = wrist3.find("body[@name='right_jaw']")
    _sub(right_jaw, "joint", name="right_jaw_slide", type="slide",
         axis=(1, 0, 0), range=(0.0, 0.045),
         damping=8.0, stiffness=0.0, armature=0.001)
    _sub(right_jaw, "inertial", pos=(0, 0, 0), mass=0.02,
         diaginertia=(1e-5, 1e-5, 1e-5))
    _sub(right_jaw, "geom", name="right_jaw_geom", type="box",
         size=(0.005, 0.012, 0.025),
         material="jaw_mat",
         friction=(1.0, 0.05, 0.001),
         contype=1, conaffinity=1, group=1)

    # TCP site at the midpoint between the open jaws — used by the IK
    # routine to target the grape position and (later) by Stage 7 as a
    # readable end-effector pose.
    _sub(wrist3, "site", name="tcp", pos=(0.0, 0.110, 0.0),
         size=(0.003,), rgba=(1.0, 0.2, 0.2, 0.6), group=4)

    # actuators — add jaw position actuators with low stiffness so the
    # commanded inward position translates into a modest contact force (a
    # gentle pinch, scaled by the LSTM force-head output)
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    _sub(act, "position", name="left_jaw_act", joint="left_jaw_slide",
         ctrlrange=(0.0, 0.045), kp=180.0, kv=10.0,
         forcerange=(-pc.GRAPE_CRUSH_THRESHOLD_N,
                     pc.GRAPE_CRUSH_THRESHOLD_N))
    _sub(act, "position", name="right_jaw_act", joint="right_jaw_slide",
         ctrlrange=(0.0, 0.045), kp=180.0, kv=10.0,
         forcerange=(-pc.GRAPE_CRUSH_THRESHOLD_N,
                     pc.GRAPE_CRUSH_THRESHOLD_N))

    # equality constraint that "welds" the grape to the gripper TCP. Disabled
    # by default; the controller activates it when the jaws are closed on
    # the grape so the pick-and-place is robust to small contact-friction
    # variations during the lift, and deactivates it once the arm is over
    # the target zone so the grape falls under gravity onto the zone.
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    _sub(eq, "weld", name="grape_grip", body1="wrist_3_link", body2="grape",
         relpose="0 0.11 0 1 0 0 0", active="false")

    # update keyframe so it includes neutral arm + open jaws
    kf = root.find("keyframe")
    if kf is None:
        kf = ET.SubElement(root, "keyframe")
        # add a default home key
    # Replace the original 6-dof home key with one that has 6 arm joints +
    # 2 jaw slides + 7 grape free joint dofs (qpos), and 8 ctrl entries.
    for child in list(kf):
        kf.remove(child)
    arm_q = list(pc.ARM_HOVER_QPOS)
    jaw_q = [0.0, 0.0]
    grape_q = [gx, gy, gz, 1.0, 0.0, 0.0, 0.0]
    qpos = arm_q + jaw_q + grape_q
    ctrl = list(pc.ARM_HOVER_QPOS) + [0.0, 0.0]
    _sub(kf, "key", name="home",
         qpos=tuple(qpos), ctrl=tuple(ctrl))

    # write -----------------------------------------------------------
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_model(scene_xml: Path = pc.SCENE_XML_OUT
               ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the scene XML on CPU and return (model, data) at the home key."""
    scene_xml = Path(scene_xml)
    if not scene_xml.exists():
        build_scene_xml(out_path=scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    # apply home keyframe
    try:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)
    return model, data


def ensure_scene() -> Path:
    """Idempotently build the canonical scene XML."""
    if not pc.SCENE_XML_OUT.exists():
        return build_scene_xml()
    return pc.SCENE_XML_OUT


if __name__ == "__main__":
    p = build_scene_xml()
    print(f"wrote {p}")
    m, d = load_model(p)
    print(f"loaded: nbody={m.nbody}  nq={m.nq}  nu={m.nu}  ncam={m.ncam}")
