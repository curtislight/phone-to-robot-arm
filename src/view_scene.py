import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("assets/two_link/my_scene.xml") #builds model
data = mujoco.MjData(model) #gets live data from model

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)