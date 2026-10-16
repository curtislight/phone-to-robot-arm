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

I film from directly above. A printed ArUco marker turns pixels into table positions in metres. The object's own position, before my hand enters and after it leaves, gives exact grab and release points. MediaPipe tracks my hand for the carried path and the timing. The tap of the object landing refines the release moment. The result is a demo: where I grabbed, the path I carried, where I let go.

That demo replays on three robots in MuJoCo. The Panda and UR5e use mink for IK. The 2-link arm uses my C++ library. A small network then learns from 10 demos to predict the path and release from the start position, tested on the other 3.

**Why a 2D arm?** The camera looks straight down, so it only sees x and y. A flat arm with a vertical slide matches that data. Height comes from the events: low at grab and release, carried in between.

**Author.** I'm [Curtis Light](https://www.linkedin.com/in/curtis-light/), third year MEng Design Engineering at Imperial College London. My background is hardware.

## Results

13 demos, each replayed on all three robots. Success means the cube's centre lands within 4cm of the target's.

| Robot | Success | Mean error to target | Mean error to my release |
|-------|---------|----------------------|--------------------------|
| Panda | 12 / 13 | 20.4 mm | 7.8 mm |
| UR5e | 10 / 13 | 24.2 mm | 2.8 mm |
| 2-link | 11 / 13 | 29.0 mm | 27.8 mm |

The last column is the one that matters. It measures how faithfully each robot copies what my hand did. The UR5e lands within 3mm of my release on average, the Panda within 8mm. Most "failures" are mine, not the robot's: on those demos my own hand released just over 4cm from the target, and the robot copied it.

The 2-link arm is the loosest. Its tool drags the cube instead of lifting it cleanly, so the cube slides. Its success rate beating the UR5e's is luck, not precision.

**Policy on the 3 held-out demos**

_Fill in after running `python src/policy.py test`._

| | Release error | Path error |
|---|---|---|
| Policy | _x mm_ | _x mm_ |
| Baseline | _x mm_ | _x mm_ |

The release baseline always places at the average spot. With a fixed target that's hard to beat. The path is where the policy should help, since it depends on where the object starts.

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
- **The object marks grab and release, the hand marks the path.** Its position in the first and last frames is exact. The finger pinch is kept as a fallback.
- **The virtual robot stands 25cm behind the tape mark**, so every demo is within reach of all three arms.
- **Fixed held-out split**: demos 04, 09 and 13, inside the spread of training positions.

## What broke

- **Meshes not found.** Included files resolve paths from the main scene, so each scene lives in its robot's folder.
- **Gripper wouldn't open.** My wait function overwrote the gripper command every frame.
- **2-link tool fell through its slide.** Without `autolimits`, MuJoCo ignores `range`.
- **pybind11 wouldn't link on macOS 27.** The linker didn't understand the newest SDK. Fixed by prefixing the build with `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.sdk`.
- **Angle conventions.** My library uses degrees from the y axis and the elbow's inside angle. MuJoCo wants radians from x and the bend from straight. Converted in `pick.py`.
- **Marker not found in 12 of 13 videos.** Glare washed the printed black out to the same brightness as the wood. Fixed by using colour instead: the table is coloured, the marker isn't, so colourful pixels get turned white before detection.
- **Release detected mid-carry.** Holding the object keeps thumb and index apart, so the pinch flickered around the threshold. Switched to tracking the object's own start and end positions.
- **Panda couldn't reach.** Most demos started too close to the robot's base for it to reach down. Moved the virtual robot back 25cm: Panda went from 4 to 12 successes out of 13.
- **2-link arm out of reach** after that move. Lengthened its links from 35 and 30cm to 40 and 35cm.

## Known issues

- The 2-link arm's tool drags the cube rather than lifting it cleanly.
- The real object was a flat sticky note standing in for the 4cm sim cube. It's also quiet, so audio missed the tap on 3 of 13 demos.
- A large shadow fell across the table while filming.
- No height data from overhead, and fingertips sit above the table, so the hand path carries a small parallax error.
- 13 demos is small. The policy is a proof of concept.

## How this was built

The scene, the robot setup in `ik.py` and `pick.py`, and the 2-link arm are mine, built while learning with Claude as a tutor. The C++ IK library is mine from before. Under the deadline, calibration, hand tracking, audio, retargeting and the policy were written with heavy help from Claude, and I've gone through each to understand it.

## Credits

[MuJoCo](https://github.com/google-deepmind/mujoco), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Panda and UR5e, Apache 2.0, I added grip sites and a suction cup), [mink](https://github.com/kevinzakka/mink), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [librosa](https://github.com/librosa/librosa). All demo data is from my own recordings.