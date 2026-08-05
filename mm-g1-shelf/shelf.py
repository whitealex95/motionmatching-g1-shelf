"""Entry frames for the pick skill.

B enters the pick clip at one of its idle (standing) frames; the ride then
plays to the end of the clip, where the vase is held.
"""
import numpy as np

import config as C


def pick_entries(lib):
    """(enter, end_of): entry frame candidates and, per entry, the last
    frame of that clip's pick ride."""
    skill = lib["skill"]
    phase = lib["phase"]
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(skill))

    enter, end_of = [], {}
    for rs, re in zip(starts, stops):
        rows = np.arange(rs, re)
        in_skill = rows[skill[rs:re] == C.SKILL_PICK]
        if len(in_skill) == 0:
            continue
        ride_end = int(in_skill[-1])
        for f in rows[phase[rs:re] == C.PHASE_IDLE]:
            enter.append(int(f))
            end_of[int(f)] = ride_end
    return np.array(enter, np.int64), end_of
