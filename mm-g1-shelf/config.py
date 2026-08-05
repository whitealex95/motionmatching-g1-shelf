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

# --- Pick skill (press B near the shelf) -----------------------------------
SKILL_LOCO, SKILL_PICK = 0, 1
PHASE_IDLE, PHASE_REACH, PHASE_GRASP, PHASE_LIFT, PHASE_HOLD = range(5)

# The whole recorded scene (shelf + vase + stance) is moved by this much,
# so the robot spawns at the origin and the shelf sits ahead of it.
SHELF_ORIGIN = [1.55, 0.0]

PICK_TRIGGER_RADIUS = 1.2   # B works within this planar distance of the vase

# B only takes when the entry query loss is below this. Measured bands:
# settled near the shelf facing the vase ~2-5, facing away ~12, still
# walking ~14+. Stance accuracy itself is checked by the grab gate.
PICK_ENTER_THRESHOLD = 10.0

# Pick search-feature block weights (one shared std per block, then these).
VASE_POS_WEIGHT = 1.0            # vase position in the robot's heading frame
HELD_WEIGHT = 0.1

# Master switch for the grab gate and the reach-phase arm IK overlay.
POST_PROCESSING = True

# The grab gate: when the ride reaches the reach phase, preview the rest of
# it from the live stance and solve the arm IK for sampled hand targets; the
# skill aborts unless every palm residual is under this.
GRAB_CHECK_SAMPLES = 6
GRAB_CHECK_TOL = 0.06

# Reach-phase arm IK overlay: DLS on the 7 right-arm joints.
IK_ITERS = 8
IK_LAMBDA = 0.05
IK_DIR_WEIGHT = 0.3
IK_STEP_CLAMP = 0.35
IK_FD_EPS = 1e-4
IK_BLEND_HALFLIFE = 0.1
