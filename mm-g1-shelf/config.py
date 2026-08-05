"""Shared constants: paths, skeleton layout, and motion-matching settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")
SHELF_DATA_DIR = os.path.join(ROOT, "data", "g1_shelf")
SHELF_META = os.path.join(SHELF_DATA_DIR, "meta.json")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
TOOLS_DIR = os.path.join(ROOT, "tools")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D): [0:3] root pos, [3:7] root quat (wxyz), [7:36] 29 joints.
# The right arm is qpos 29..35 (joint dof index 22..28).
JOINTS = slice(7, 36)
ARM_QPOS = slice(29, 36)
ARM_DOF = slice(22, 29)

FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
HAND_BODY = "right_wrist_yaw_link"
PALM_OFFSET = [0.10, -0.007, 0.0]     # palm point in the wrist frame

# --- Motion-matching core (same as motionmatching-g1-door) -----------------
HORIZONS = [10, 20, 30]
SEARCH_TIME = 0.15
INERT_HALFLIFE = 0.075
VEL_HALFLIFE = 0.2
ROT_HALFLIFE = 0.2
CURRENT_BIAS = 0.01
APPROX_BIAS = 0.01

ROOT_POS_SMOOTH = 15
ROOT_DIR_SMOOTH = 31

CLIP_TRIM = {
    "walk1_subject5":            (80, 7759),
    "run1_subject5":             (86, 7068),
    "pushAndStumble1_subject5":  (198, 353),
}

LIB_VERSION = 2                # v2: dropped shelf_dir from the library
SEARCH_TAIL = HORIZONS[-1]

CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]
MIRROR = True

MAX_SPEED = 2.5
WALK_SCALE = 0.4

# --- Pick skill (press B) --------------------------------------------------
SKILL_LOCO, SKILL_PICK = 0, 1
PHASE_IDLE, PHASE_REACH, PHASE_GRASP, PHASE_LIFT, PHASE_HOLD = range(5)

# The whole recorded scene (shelf + vase + stance) is moved by this much,
# so the robot spawns at the origin and the shelf sits ahead of it.
SHELF_ORIGIN = [1.55, 0.0]

# Move-to-pick: B routes the robot onto the rail (the line through the
# stance along its heading) 0.6 m behind the stance, then straight in.
# The rail aims MOVE_OVERSHOOT past the stance, so the commanded walk never
# slows into the dead zone; the pick starts when the robot crosses the
# stance plane, still walking.
MOVE_OVERSHOOT = 0.35       # the rail target sits this far past the stance (m)
MOVE_ARRIVE_NEAR = 0.12     # this close counts as arrived right away (m)
MOVE_ARRIVE_YAW = 0.6       # close enough to the stance heading (rad)
MOVE_TIMEOUT = 8.0          # give up walking after this long (s)

# During move-to-pick the future taps are read straight off the planned
# route (walk the remaining path at the approach speed and sample it).
# Whatever offset is left when the grab happens is absorbed by the vase snap.

# Path snap: inside this radius of the stance the root cannot leave the
# rail -- the cross-track part of the matched motion is projected out at
# the given half-life (fast: gone within ~0.3 s).
SNAP_RADIUS = 4.0
SNAP_HALFLIFE = 2.0
