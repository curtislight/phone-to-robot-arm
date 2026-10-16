'''Pick and place using the IK tools from ik.py'''

import time
import mujoco
import mujoco.viewer
import numpy as np
import ik2d

from ik import go_to, model, data, cube_pos, target_pos, dt, steps_per_frame, hover, fps, max_q, suction_height, robot, frames

if robot == 1:
    OPEN = 255
    CLOSED = 0
    go = go_to
    above = np.array([0, 0, 0.1])  # 10cm above, for approaching and lifting

elif robot == 2:
    OPEN = 0
    CLOSED = 1
    go = go_to
    above = np.array([0, 0, 0.1])  # 10cm above, for approaching and lifting

elif robot == 3:
    OPEN = 0
    CLOSED = 1
    above = np.array([0, 0, 0.04])  # 4cm above, the slide only travels 6cm
    l1 = model.body("link2").pos[0]  # link 1 length, read from the XML
    l2 = model.body("tool").pos[0]   # link 2 length

    def go(position, viewer):
        start = data.site("grip").xpos.copy()  # where the cup is now

        for i in range(frames):
            s = (i + 1) / frames                     # how far along the path, 0 to 1
            point = start + s * (position - start)   # this frame's point on the path

            if ik2d.is_possible(l1, l2, point[0], point[1]):
                alpha, beta, side = ik2d.solve(l1, l2, point[0], point[1])

                shoulder = np.radians(90 - alpha)   # library measures from y, MuJoCo from x
                bend = 180 - beta                    # library gives the inside angle, MuJoCo wants bend from straight
                elbow = np.radians(-bend if side == 0 else bend)

                data.ctrl[0] = shoulder
                data.ctrl[1] = elbow
                data.ctrl[2] = np.clip(point[2] - 0.08, -0.06, 0)  # cup face sits at 8cm when lift is 0

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

            if not viewer.is_running():
                return


def wait(seconds, viewer):  # lets actuators grip before moving
    frames = int(seconds * fps)
    for i in range(frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        if not viewer.is_running():
            return


with mujoco.viewer.launch_passive(model, data) as viewer:
    go(cube_pos + above + suction_height, viewer)   # hover above the cube first
    go(cube_pos + suction_height, viewer)           # then come straight down onto it
    data.ctrl[max_q] = CLOSED
    wait(0.75, viewer)
    go(cube_pos + above + suction_height, viewer)
    go(target_pos + hover + suction_height, viewer)
    data.ctrl[max_q] = OPEN
    wait(0.75, viewer)
    go(target_pos + above + suction_height, viewer)

    # keep window open
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)