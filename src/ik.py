# Inverse kinematics solver using mink
import time
import mujoco
import mujoco.viewer
import mink
from mink import SE3, solve_ik
import numpy as np

model = mujoco.MjModel.from_xml_path("assets/franka_emika_panda/my_scene.xml")
data = mujoco.MjData(model)

home_id = model.key("home").id  # gets home joint angles from panda.xml
mujoco.mj_resetDataKeyframe(model, data, home_id)
mujoco.mj_forward(model, data)

configuration = mink.Configuration(model)   #configures mink model
configuration.update(data.qpos)

# Gripper points down in home, so save that orientation and reuse it for every target
rotation = configuration.get_transform_frame_to_world("grip", "site").rotation()

#sets time and fps of simulation
duration, fps = 5.0, 60
frames = int(duration * fps)
gain = 1.0 - 0.01 ** (1.0 / frames)  #makes smooth translation

dt = 1.0 / fps
steps_per_frame = int(dt / model.opt.timestep)  # physics runs at 500Hz, IK at 60Hz, so ~8 sim steps per IK frame

grip_task = mink.FrameTask(     
    frame_name="grip",
    frame_type="site",
    position_cost=1.0,  #position accuracy is important
    orientation_cost=0.5,   
    lm_damping=1.0,
    gain = gain
)

#gets cube and target positions
# .copy() freezes them, otherwise they'd keep updating as the sim runs
cube_pos = data.body("cube").xpos.copy()
target_pos = data.geom("target").xpos.copy()

hover = np.array([0, 0, 0.03])  # 3cm above, so the hand doesn't hit the table

def go_to(position, viewer):
    # Bundle position with the pointing-down rotation into a full pose
    grip_task.set_target(SE3.from_rotation_and_translation(rotation, position))

    for i in range(frames):
        vel = solve_ik(configuration, [grip_task], dt, "daqp")
        configuration.integrate_inplace(vel, dt)

        data.ctrl[:7] = configuration.q[:7]  # send the 7 arm joint angles to the motors

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)

        if not viewer.is_running():
            return

with mujoco.viewer.launch_passive(model, data) as viewer:
    go_to(cube_pos, viewer)
    go_to(target_pos + hover, viewer)

    # keeps the window open after the moves
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)