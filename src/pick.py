'''Pick and place using the IK tools from ik.py'''

import time
import mujoco
import mujoco.viewer
import numpy as np

from ik import go_to, model, data, cube_pos, target_pos, dt, steps_per_frame, hover, fps, grip_task, configuration, solve_ik

OPEN = 255
CLOSED = 0
above = np.array([0, 0, 0.1])  # 10cm above, for approaching and lifting


def wait(seconds, viewer):
    frames = int(seconds * fps)
    for i in range(frames):
    
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    
            if not viewer.is_running():
                return


with mujoco.viewer.launch_passive(model, data) as viewer:
    go_to(cube_pos, viewer)
    data.ctrl[7] = CLOSED
    wait(0.75, viewer)
    go_to(cube_pos + above, viewer)
    go_to(target_pos + hover, viewer)
    data.ctrl[7] = OPEN
    wait(0.75, viewer)
    go_to(target_pos + above, viewer)
    
    # keep window open
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)