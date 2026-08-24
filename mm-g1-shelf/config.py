"""Shared constants: paths, skeleton layout, and motion-matching settings."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")
SHELF_DATA_DIR = os.path.join(ROOT, "data", "g1_shelf")
SHELF_META = os.path.join(SHELF_DATA_DIR, "meta.json")
PICK_SOURCE = os.path.join(SHELF_DATA_DIR, "pick2.npz")
PICK_QPOS = os.path.join(SHELF_DATA_DIR, "pick2_g1.npz")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
TOOLS_DIR = os.path.join(ROOT, "tools")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D): [0:3] root pos, [3:7] root quat (wxyz), [7:36] 29 joints.
ARM_QPOS = slice(29, 36)              # the right arm
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
PALM_OFFSET = [0.10, -0.007, 0.0]     # palm point in the right wrist frame

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

LIB_VERSION = 4                # bump when the library build changes

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

# Move-to-pick: B walks a planned route to the recorded stance -- straight
# to a way-in point 0.6 m behind it, around a rounded corner, then in along
# the stance heading (the rail). The future taps are read straight off this
# route. The rail aims MOVE_OVERSHOOT past the stance, so the commanded
# walk never slows into the dead zone; the pick starts when the robot
# crosses the stance plane, still walking. Whatever offset is left at the
# grab is absorbed by the vase snap.
MOVE_OVERSHOOT = 0.35       # the rail target sits this far past the stance (m)
MOVE_ARRIVE_NEAR = 0.12     # this close counts as arrived right away (m)
MOVE_ARRIVE_YAW = 0.6       # close enough to the stance heading (rad)
MOVE_TIMEOUT = 8.0          # give up walking after this long (s)

# Path snap: on the final leg, inside this radius of the stance, the root
# cannot leave the rail -- the cross-track and yaw parts of the matched
# motion are projected out at this half-life.
SNAP_RADIUS = 4.0
SNAP_HALFLIFE = 1.0
