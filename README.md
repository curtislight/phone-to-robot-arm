# phone-to-robot-arm

Film yourself moving a cube with your phone. Three different robots do the same in simulation.

Built for Humanoid's Robot Learning intern challenge, October 2026.

## Highlights

- **One demo, three bodies.** A Franka Panda, a UR5e, and a 2-link arm I designed.
- **My own C++ in the loop.** The 2-link arm runs on my 2D IK library, unmodified, called through pybind11.
- **Vision plus sound.** MediaPipe tracks the hand. The tap of the cube landing sharpens the release timing.
- **A small learned policy**, tested on held-out demos against baselines.
- **Everything is scored.** Every replay logs how far the cube landed from the target.

## Overview

Pick up a cube, put it on a target. Simple task, clear success metric, clear release moment.

I film from directly above. A printed ArUco marker turns pixels into table positions in metres. MediaPipe tracks my hand, and the gap between thumb and index finger gives the grab and release. The tap of the cube landing refines the release. The result is a demo: where I grabbed, the path I carried, where I let go.

That demo replays on three robots in MuJoCo. The Panda and UR5e use mink for IK. The 2-link arm uses my C++ library. A small network then learns from 14 demos to predict the path and release from the cube's start position, tested on the other 4.

**Why a 2D arm?** The camera looks straight down, so it only sees x and y. A flat arm with a vertical slide matches that data. Height comes from the events: low at grab and release, carried in between.

**Author.** I'm [Curtis Light](https://www.linkedin.com/in/curtis-light/), third year MEng Design Engineering at Imperial College London. My background is hardware. This project was me learning robot software: MuJoCo, IK, computer vision, pybind11.

## Results

_Fill in after running._

| Robot | Success | Mean error |
|-------|---------|------------|
| Panda | _x / 18_ | _x mm_ |
| UR5e | _x / 18_ | _x mm_ |
| 2-link | _x / 18_ | _x mm_ |

| Held-out | Release error | Path error |
|---|---|---|
| Policy | _x mm_ | _x mm_ |
| Baseline | _x mm_ | _x mm_ |

The release baseline always places at the average spot. With a fixed target that's hard to beat. The path is where the policy should help, since it depends on where the cube starts.

## Usage

```
python src/calibrate.py videos/demo_01.mov click     # check calibration
python src/hand_track.py videos/*.mov                # videos to demos
python src/audio.py videos/*.mov                     # refine release
mjpython src/retarget.py demos/demo_01.json          # watch it: 1 Panda, 2 UR5e, 3 2-link
python src/policy.py train
python src/policy.py test
```

All demos on one robot, no window:

```
for f in demos/demo_*.json; do echo 1 | python src/retarget.py $f --headless; done
```

## Installation

Python 3.12, tested on macOS.

```
git clone --recurse-submodules https://github.com/curtislight/phone-to-robot-arm.git
cd phone-to-robot-arm
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir models && curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Build the C++ bridge:

```
c++ -O3 -shared -std=c++17 -undefined dynamic_lookup $(python -m pybind11 --includes) cpp/bindings.cpp cpp/ik_solver/solver.cpp cpp/ik_solver/vec2.cpp -o src/ik2d$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
```

On macOS the viewer needs `mjpython`.

## Design choices

- **Suction on the UR5e and 2-link arm**, via MuJoCo's adhesion actuator. Menagerie's UR5e has no gripper, and real ones often use suction.
- **One sequence, three robots.** `pick.py` writes the steps once. Each robot brings its own `go` function and gripper values.
- **Link lengths read from the XML**, so the model is the single source of truth.
- **Adaptive grasp threshold**: halfway between the most closed and most open hand in each video.
- **Fixed held-out split**: demos 04, 09, 13, 17, inside the spread of training positions.

## What broke

- **Meshes not found.** Included files resolve paths from the main scene, so each scene lives in its robot's folder.
- **Gripper wouldn't open.** My wait function overwrote the gripper command every frame.
- **2-link tool fell through its slide.** Without `autolimits`, MuJoCo ignores `range`.
- **pybind11 wouldn't link on macOS 27.** The linker didn't understand the newest SDK. Fixed by prefixing the build with `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`.
- **Angle conventions.** My library uses degrees from the y axis and the elbow's inside angle. MuJoCo wants radians from x and the bend from straight. Converted in `pick.py`.

## Known issues

- The round suction cup can twist the cube on the 2-link arm.
- No height data from overhead, and fingertips sit above the table, so positions carry a small parallax error.
- 18 demos is small. The policy is a proof of concept.

## How this was built

The scene, robot setup in `ik.py` and `pick.py`, and the 2-link arm are mine, built while learning with Claude as a tutor. The C++ IK library is mine from before. Under the deadline, calibration, hand tracking, audio, retargeting and the policy were written with heavy help from Claude, and I've gone through each to understand it.

## Credits

[MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Panda and UR5e, Apache 2.0, I added grip sites and a suction cup), [mink](https://github.com/kevinzakka/mink), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [librosa](https://github.com/librosa/librosa). All demo data is from my own recordings.