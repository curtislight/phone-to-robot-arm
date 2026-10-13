'''Pick and place using the IK tools from ik.py'''

import time
import mujoco
import mujoco.viewer
import numpy as np

from ik import go_to, model, data, cube_pos, target_pos, dt, steps_per_frame, hover, fps, max_q, suction_height, robot

if robot == 1:
    OPEN = 255
    CLOSED = 0

elif robot == 2:
    OPEN = 0
    CLOSED = 1
     
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
    go_to(cube_pos + suction_height, viewer)
    data.ctrl[max_q] = CLOSED
    wait(0.75, viewer)
    go_to(cube_pos + above + suction_height, viewer)
    go_to(target_pos + hover + suction_height, viewer)
    data.ctrl[max_q] = OPEN
    wait(0.75, viewer)
    go_to(target_pos + above + suction_height, viewer)
    
    # keep window open
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)