'''Replays a demo on the selected robot and measures how well it went
Pick the robot when asked: 1 Panda, 2 UR5e, 3 two-link.
'''

import csv
import json
import sys
import time
from pathlib import Path
 
import numpy as np
 
RECORD = sys.argv[sys.argv.index("--record") + 1] if "--record" in sys.argv else None
HEADLESS = "--headless" in sys.argv or RECORD is not None
if HEADLESS:
    time.sleep = lambda seconds: None  # no real-time pacing when nobody's watching
 
import mujoco
import mujoco.viewer
 
import config
from ik import model, data, configuration, robot, suction_height, hover, max_q
from pick import go, wait, OPEN, CLOSED, above
 
ROBOT_NAMES = {1: "panda", 2: "ur5e", 3: "two_link"}
CUBE_HALF = 0.02          # cube is 4cm, so its centre sits 2cm above the table
CARRY_WAYPOINTS = 3       # how many points of the human's path the robot passes through
SUCCESS_RADIUS = 0.04     # cube centre within 4cm of the target centre, i.e. on the green square
 
 
class NoViewer:
    '''Stands in for the viewer in headless mode, so go and wait work unchanged.'''
    def sync(self):
        pass
 
    def is_running(self):
        return True
 
 
class Recorder(NoViewer):
    '''Stands in for the viewer and renders a frame every few steps, off screen.'''
    def __init__(self, every=10, width=480, height=360):
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0.45, 0.0, 0.1]  # middle of the work area
        self.camera.distance = 1.5
        self.camera.azimuth = 150
        self.camera.elevation = -30
        self.every = every
        self.count = 0
        self.frames = []
 
    def sync(self):
        self.count += 1
        if self.count % self.every == 0:  # every 10th frame, so the clip plays about 6x real speed
            self.renderer.update_scene(data, camera=self.camera)
            self.frames.append(self.renderer.render().copy())
 
    def save(self, path, fps=30):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".mp4":
            import imageio_ffmpeg
            height, width = self.frames[0].shape[:2]
            writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, macro_block_size=1)
            writer.send(None)  # starts the writer
            for frame in self.frames:
                writer.send(frame)
            writer.close()
        else:
            from PIL import Image
            images = [Image.fromarray(frame) for frame in self.frames]
            images[0].save(path, save_all=True, append_images=images[1:],
                           duration=int(1000 / fps), loop=0, optimize=True)
        print(f"Saved {path} ({len(self.frames)} frames)")
 
 
def place_scene(demo):
    '''Puts the cube where the human grabbed it, and the target where config says it is.'''
    cube_joint = model.body("cube").jntadr[0]
    qpos = model.jnt_qposadr[cube_joint]
    qvel = model.jnt_dofadr[cube_joint]
 
    x, y = demo["grasp_xy"]
    data.qpos[qpos:qpos + 3] = [x, y, CUBE_HALF]  # position
    data.qpos[qpos + 3:qpos + 7] = [1, 0, 0, 0]   # rotation: sitting flat, no twist
    data.qvel[qvel:qvel + 6] = 0                   # not moving
 
    model.geom_pos[model.geom("target").id][:2] = config.TARGET_CENTER
 
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos)  # keep mink's copy in sync
 
 
def on_table(xy):
    '''A table position at cube-centre height.'''
    return np.array([xy[0], xy[1], CUBE_HALF])
 
 
def run(demo, viewer):
    grasp = on_table(demo["grasp_xy"])
    release = on_table(demo["release_xy"])
    path = demo["path_xy"]
 
    # a few evenly spaced points from the middle of the human's path, skipping the two ends
    picks = np.linspace(0, len(path) - 1, CARRY_WAYPOINTS + 2)[1:-1].round().astype(int)
    carry = [on_table(path[i]) + above for i in picks]
 
    go(grasp + above + suction_height, viewer)      # hover over the cube
    go(grasp + suction_height, viewer)              # come down onto it
    data.ctrl[max_q] = CLOSED
    wait(0.75, viewer)
    go(grasp + above + suction_height, viewer)      # lift
 
    for point in carry:                             # follow the human's path
        go(point + suction_height, viewer)
 
    go(release + above + suction_height, viewer)    # over the release point
    go(release + hover + suction_height, viewer)    # lower
    data.ctrl[max_q] = OPEN
    wait(0.75, viewer)
    go(release + above + suction_height, viewer)    # move away
    wait(1.0, viewer)                               # let the cube settle before measuring
 
    cube_xy = data.body("cube").xpos[:2].copy()
    return {
        "demo": demo.get("name", "unknown"),
        "source": demo.get("source", "human"),
        "robot": ROBOT_NAMES[robot],
        "error_to_target_mm": round(float(np.linalg.norm(cube_xy - np.array(config.TARGET_CENTER))) * 1000, 1),
        "error_to_release_mm": round(float(np.linalg.norm(cube_xy - np.array(demo["release_xy"]))) * 1000, 1),
        "success": bool(np.linalg.norm(cube_xy - np.array(config.TARGET_CENTER)) < SUCCESS_RADIUS),
    }
 
 
def save_result(result, path=Path("results/results.csv")):
    path.parent.mkdir(exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(result)
 
 
if __name__ == "__main__":
    files = [arg for arg in sys.argv[1:] if not arg.startswith("--") and arg != RECORD]
    if not files:
        print(__doc__)
        sys.exit(1)
 
    demo = json.loads(Path(files[0]).read_text())
    place_scene(demo)
 
    if RECORD:
        recorder = Recorder()
        result = run(demo, recorder)
        recorder.save(RECORD)
    elif HEADLESS:
        result = run(demo, NoViewer())
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            result = run(demo, viewer)
            while viewer.is_running():  # keep the window open to look around
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)
 
    if not RECORD:  # recordings are re-runs for media, don't double-count them in the results
        save_result(result)
    print(result)
 