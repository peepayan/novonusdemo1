"""DB2 restimulus label -> coarse 5-class intent mapping.

Source taxonomy
---------------
DB2 movement numbering used in this dataset (restimulus values):
  0                  rest / gap
  1..17  (block E1)  Exercise B — basic movements of fingers and wrist
  18..40 (block E2)  Exercise C — 23 grasping and functional movements
  41..49 (block E3)  Exercise D — 9 force-task movements (NOT included in training)

Five coarse intent classes used for the LSTM classification head:
  0 REST          quiescence
  1 REACHING      wrist/forearm articulation, arm movement
  2 GRIPPING      power grasps, closed fist, whole-hand power closures
  3 STABILIZING   precision / pinch / tripod / lateral / fine-motor grasps
  4 RELEASING    hand opening, finger abduction, finger extension

Mapping rationale (per Ninapro DB2 movement list and Cutkosky / Feix grasp
taxonomy):

  EXERCISE B  (E1, labels 1..17)
    1  thumb up                                 -> RELEASING   (single-finger extension)
    2  index+middle extend, others flex         -> STABILIZING (mixed configuration / fine)
    3  ring+little flex, others extend          -> STABILIZING (mixed configuration / fine)
    4  thumb opposes little finger              -> STABILIZING (precision opposition)
    5  abduction of all fingers                 -> RELEASING   (hand spread / open)
    6  fingers flexed together in fist          -> GRIPPING    (power closure)
    7  pointing index                           -> STABILIZING (precision pointing)
    8  adduction of extended fingers            -> STABILIZING (fine motor, fingers held extended)
    9  wrist supination (middle-finger axis)    -> REACHING
   10  wrist pronation  (middle-finger axis)    -> REACHING
   11  wrist supination (little-finger axis)    -> REACHING
   12  wrist pronation  (little-finger axis)    -> REACHING
   13  wrist flexion                            -> REACHING
   14  wrist extension                          -> REACHING
   15  wrist radial deviation                   -> REACHING
   16  wrist ulnar deviation                    -> REACHING
   17  wrist extension with closed hand         -> REACHING   (dominant motion is wrist)

  EXERCISE C  (E2, labels 18..40)
   18  large diameter grasp                     -> GRIPPING
   19  small diameter grasp                     -> GRIPPING
   20  fixed hook grasp                         -> GRIPPING   (whole-hand hook power)
   21  index extension grasp                    -> GRIPPING   (power grip + extended index)
   22  medium wrap                              -> GRIPPING
   23  ring grasp                               -> STABILIZING (precision ring)
   24  prismatic four-finger grasp              -> STABILIZING (precision prismatic)
   25  stick grasp                              -> GRIPPING    (cylindrical power)
   26  writing tripod grasp                     -> STABILIZING (precision tripod)
   27  power sphere grasp                       -> GRIPPING
   28  three-finger sphere grasp                -> STABILIZING (precision sphere, 3-finger)
   29  precision sphere grasp                   -> STABILIZING
   30  tripod grasp                             -> STABILIZING
   31  prismatic pinch grasp                    -> STABILIZING
   32  tip pinch grasp                          -> STABILIZING
   33  quadpod grasp                            -> STABILIZING
   34  lateral grasp (key grip)                 -> STABILIZING (precision lateral)
   35  parallel extension grasp                 -> STABILIZING (extended-finger precision)
   36  extension type grasp                     -> STABILIZING (extended-finger precision)
   37  power disk grasp                         -> GRIPPING
   38  open a bottle with tripod grasp          -> STABILIZING (dominant grasp = tripod)
   39  turn a screw                             -> REACHING   (dominant motion = wrist rotation)
   40  cut something                            -> REACHING   (dominant motion = arm)

Ambiguous labels (resolved with the rationale above, surfaced here for honesty):
   1   thumb up           — could read as STABILIZING (isolated finger posture); chose
                            RELEASING because the action is an *extension* of the thumb
                            from rest, matching the releasing class definition.
   8   adduction extended — fingers stay extended (release-like) but the motion is
                            closing fingers together (stab-like). Chose STABILIZING
                            since the action is fine-motor adduction, not hand opening.
   17  wrist ext + closed hand — has a grip component; chose REACHING because the
                            dynamic component is the wrist extension.
   21  index extension grasp  — borderline gripping/stabilizing; chose GRIPPING
                            because the underlying hold is a power wrap.
   25  stick grasp        — narrow cylinder; chose GRIPPING (Cutkosky cylindrical-power).
   35,36 extension grasps — extended fingers but grasp-shaped; chose STABILIZING.
   38  open a bottle      — composite; mapped to the dominant tripod grasp -> STABILIZING.
   39,40 turn screw / cut — composite functional moves; mapped to dominant
                            wrist/arm motion -> REACHING.
"""

from __future__ import annotations

INTENT_NAMES: tuple[str, ...] = (
    "REST",
    "REACHING",
    "GRIPPING",
    "STABILIZING",
    "RELEASING",
)
N_INTENT_CLASSES: int = len(INTENT_NAMES)

# label -> intent class id
DB2_LABEL_TO_INTENT: dict[int, int] = {
    0:  0,  # rest

    # Exercise B / E1
    1:  4,
    2:  3,
    3:  3,
    4:  3,
    5:  4,
    6:  2,
    7:  3,
    8:  3,
    9:  1,
    10: 1,
    11: 1,
    12: 1,
    13: 1,
    14: 1,
    15: 1,
    16: 1,
    17: 1,

    # Exercise C / E2
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 2,
    23: 3,
    24: 3,
    25: 2,
    26: 3,
    27: 2,
    28: 3,
    29: 3,
    30: 3,
    31: 3,
    32: 3,
    33: 3,
    34: 3,
    35: 3,
    36: 3,
    37: 2,
    38: 3,
    39: 1,
    40: 1,
}

# Free-text label description used in the printed mapping table.
DB2_LABEL_DESCRIPTION: dict[int, str] = {
    0:  "rest",
    1:  "thumb up",
    2:  "index+middle extend, ring+little+thumb flex",
    3:  "ring+little flex, others extend",
    4:  "thumb opposes little finger",
    5:  "abduction of all fingers",
    6:  "fingers flexed in a fist",
    7:  "pointing index",
    8:  "adduction of extended fingers",
    9:  "wrist supination (middle axis)",
    10: "wrist pronation (middle axis)",
    11: "wrist supination (little axis)",
    12: "wrist pronation (little axis)",
    13: "wrist flexion",
    14: "wrist extension",
    15: "wrist radial deviation",
    16: "wrist ulnar deviation",
    17: "wrist extension with closed hand",
    18: "large diameter grasp",
    19: "small diameter grasp",
    20: "fixed hook grasp",
    21: "index extension grasp",
    22: "medium wrap",
    23: "ring grasp",
    24: "prismatic four-finger grasp",
    25: "stick grasp",
    26: "writing tripod grasp",
    27: "power sphere grasp",
    28: "three-finger sphere grasp",
    29: "precision sphere grasp",
    30: "tripod grasp",
    31: "prismatic pinch grasp",
    32: "tip pinch grasp",
    33: "quadpod grasp",
    34: "lateral grasp (key grip)",
    35: "parallel extension grasp",
    36: "extension type grasp",
    37: "power disk grasp",
    38: "open a bottle with tripod grasp",
    39: "turn a screw",
    40: "cut something",
}

AMBIGUOUS_LABELS: dict[int, str] = {
    1:  "could be STABILIZING; chose RELEASING (thumb extension from rest)",
    8:  "adduction with extended fingers; chose STABILIZING (fine motor)",
    17: "wrist motion + closed hand; chose REACHING (dominant = wrist)",
    21: "borderline gripping/stabilizing; chose GRIPPING (underlying power wrap)",
    25: "narrow cylinder; chose GRIPPING (Cutkosky cylindrical-power)",
    35: "extended-finger grasp; chose STABILIZING",
    36: "extended-finger grasp; chose STABILIZING",
    38: "composite; mapped to dominant tripod grasp -> STABILIZING",
    39: "composite functional; mapped to dominant wrist rotation -> REACHING",
    40: "composite functional; mapped to dominant arm motion -> REACHING",
}


def intent_id(label: int) -> int:
    """Map a single DB2 restimulus label to its 5-class intent id."""
    return DB2_LABEL_TO_INTENT.get(int(label), 0)


def build_label_array_mapping(max_label: int = 40) -> list[int]:
    """Return a flat list ``intent[label]`` for vectorised lookup with numpy.

    Use as: ``intent = np.asarray(build_label_array_mapping())[labels]``
    """
    return [DB2_LABEL_TO_INTENT.get(i, 0) for i in range(max_label + 1)]


def format_table() -> str:
    """Build a human-readable label -> intent table."""
    lines = []
    header = f"{'label':>5} | {'intent':<12} | description"
    lines.append(header)
    lines.append("-" * len(header))
    for lbl in sorted(DB2_LABEL_TO_INTENT):
        iid = DB2_LABEL_TO_INTENT[lbl]
        desc = DB2_LABEL_DESCRIPTION.get(lbl, "")
        lines.append(f"{lbl:>5} | {INTENT_NAMES[iid]:<12} | {desc}")
    lines.append("")
    lines.append("ambiguous labels (resolved as documented):")
    for lbl, note in AMBIGUOUS_LABELS.items():
        lines.append(f"  {lbl:>3}: {note}")
    return "\n".join(lines)


def mapping_json_dict() -> dict:
    """Serializable mapping object for outputs/stage2/class_mapping.json."""
    return {
        "intent_names": list(INTENT_NAMES),
        "n_intent_classes": N_INTENT_CLASSES,
        "label_to_intent_id": {str(k): v for k, v in DB2_LABEL_TO_INTENT.items()},
        "label_description":  {str(k): v for k, v in DB2_LABEL_DESCRIPTION.items()},
        "ambiguous": {str(k): v for k, v in AMBIGUOUS_LABELS.items()},
        "notes": [
            "Mapping derived from Ninapro DB2 movement list (Exercise B labels 1-17,",
            "Exercise C labels 18-40) and the Cutkosky/Feix grasp taxonomy.",
            "Block E3 (labels 41-49, force-task movements) is excluded from training.",
        ],
    }


if __name__ == "__main__":
    print(format_table())
