# phone-to-robot-arm

Film yourself doing a task on your phone. Pull the hand motion out of the video. Replay it on robot arms in simulation.

Built for Humanoid's Robot Learning intern challenge. Still in progress.

## The idea

One human, many bodies. A single phone demo should drive three very different robots: a Franka Panda, a UR5e, and a small 2-link arm running my own C++ inverse kinematics library. If the same motion transfers to all three, the retargeting works.

The task is simple on purpose: pick up a cube, place it on a target. Clear success metric, clear release moment.

## Where it's at

Done
[x] MuJoCo scene with table, cube and target
[x] IK with mink, gripper kept pointing down, smooth moves
[x] Scripted pick and place on the Panda (parallel gripper)
[x] Same pipeline on the UR5e, with a suction gripper
[x] One codebase, pick the robot at launch

Next
[] Film 18 overhead demos, 4 held out for testing
[] ArUco marker calibration, pixels to table coordinates
[] MediaPipe hand tracking for grasp and release
[] Audio onset detection, timing the release from the sound of the cube landing
[] 2-link arm driven by my C++ IK library
[] Small behaviour cloning policy, tested on the held-out positions

## Run it

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install mujoco mink numpy daqp
    mjpython src/pick.py

Type 1 for the Panda, 2 for the UR5e. On macOS the viewer needs `mjpython` instead of `python`.

## Changes to borrowed models

- `panda.xml`: added a `grip` site between the fingertips
- `ur5e.xml`: added a suction cup and `grip` site on the wrist flange
- UR5e suction uses MuJoCo's adhesion actuator, added in my scene file

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco), Google DeepMind
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for the Panda and UR5e models (Apache 2.0)
- [mink](https://github.com/kevinzakka/mink), Kevin Zakka, for IK