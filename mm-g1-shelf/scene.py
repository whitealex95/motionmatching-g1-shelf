"""The interactive scene: kinematic G1 + an IKEA shelf and bottle.

Built through tools/shelf_model.py. Nothing is simulated: the robot qpos
comes from the motion matching output, and the bottle pose comes from the
matcher (on the shelf, or stuck to the palm).
"""
import json
import sys
import numpy as np
import mujoco

import config as C

sys.path.insert(0, C.TOOLS_DIR)
import shelf_model as SM


class ShelfScene:
    def __init__(self, scene_xml=C.SCENE_XML):
        with open(C.SHELF_META) as f:
            meta = json.load(f)["shelf"]
        self.model, self.ids = SM.build_model(scene_xml, meta,
                                              origin_xy=C.SHELF_ORIGIN)
        self.data = mujoco.MjData(self.model)
        self.vase_rest = self.data.mocap_pos[self.ids["vase_mocap"]].copy()
        self.held = False
        self.reset()

    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.data.mocap_pos[self.ids["vase_mocap"]] = self.vase_rest
        self.data.mocap_quat[self.ids["vase_mocap"]] = [1.0, 0.0, 0.0, 0.0]
        self.set_held(False)
        mujoco.mj_forward(self.model, self.data)

    def set_held(self, held):
        if held != self.held:
            self.held = held
            SM.set_vase_color(self.model, self.ids, held)

    def step(self, robot_qpos, vase_pos, vase_quat, held):
        """One 30 Hz frame: place the robot and the vase, no physics."""
        self.data.qpos[0:36] = robot_qpos
        self.data.mocap_pos[self.ids["vase_mocap"]] = vase_pos
        self.data.mocap_quat[self.ids["vase_mocap"]] = vase_quat
        self.set_held(held)
        mujoco.mj_forward(self.model, self.data)
