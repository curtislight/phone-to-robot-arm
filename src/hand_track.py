'''hand tracking

Reads an overhead demo video, finds the hand in every frame with MediaPipe,
and works out when the cube was grabbed, the path it was carried along, and
where it was put down. Positions are converted to table coordinates (metres)
with the marker calibration from calibrate.py.
'''

import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from calibrate import calibrate, pixel_to_table

MODEL_PATH = "models/hand_landmarker.task"

# MediaPipe numbers its 21 hand points. These are the ones used here.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_KNUCKLE = 0, 4, 8, 9

PATH_POINTS = 10     # the carried path is resampled to this many points
SMOOTH_FRAMES = 5    # moving average window for the pinch signal (odd number)
HOLD_FRAMES = 5      # a pinch has to last this many frames to count


def make_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,  # VIDEO mode follows the hand from frame to frame
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmark_px(hand, index, width, height):
    '''MediaPipe gives positions as fractions of the image size. This converts one to pixels.'''
    return np.array([hand[index].x * width, hand[index].y * height])


def track_video(video_path, debug_path=None):
    '''Runs MediaPipe on every frame.
    Returns the grasp point (pixels) and pinch ratio for each frame, NaN where no hand was found.'''
    video = cv2.VideoCapture(str(video_path))
    fps = video.get(cv2.CAP_PROP_FPS)
    points, pinches = [], []
    writer = None

    with make_landmarker() as landmarker:
        frame_index = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR, MediaPipe wants RGB
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(frame_index * 1000 / fps))

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                thumb = landmark_px(hand, THUMB_TIP, width, height)
                index = landmark_px(hand, INDEX_TIP, width, height)
                hand_size = np.linalg.norm(landmark_px(hand, WRIST, width, height)
                                           - landmark_px(hand, MIDDLE_KNUCKLE, width, height))
                point = (thumb + index) / 2                        # between the fingertips, where the cube is held
                pinch = np.linalg.norm(thumb - index) / hand_size  # divided by hand size so camera distance doesn't matter
            else:
                point = np.array([np.nan, np.nan])
                pinch = np.nan

            points.append(point)
            pinches.append(pinch)

            if debug_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(debug_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                if not np.isnan(pinch):
                    cv2.circle(frame, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"pinch {pinch:.2f}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                writer.write(frame)

            frame_index += 1

    video.release()
    if writer:
        writer.release()
    return np.array(points), np.array(pinches), fps


def fill_gaps(values):
    '''Fills frames where the hand wasn't found with a straight line between the frames either side.'''
    values = np.asarray(values, dtype=float)
    frames = np.arange(len(values))
    found = ~np.isnan(values)
    if found.sum() < 2:
        raise RuntimeError("Hand barely found in this video. Check lighting and that the hand stays in shot")
    return np.interp(frames, frames[found], values[found])


def smooth(values, window=SMOOTH_FRAMES):
    '''Moving average, so a single jittery frame can't trigger a fake grasp or release.'''
    padded = np.pad(values, window // 2, mode="edge")  # repeat the end values so the edges aren't dragged down
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def find_grasp_and_release(pinch):
    '''Grasp: first time the pinch stays closed for HOLD_FRAMES.
    Release: first time after that it stays open for HOLD_FRAMES.'''
    threshold = (pinch.min() + pinch.max()) / 2  # halfway between most closed and most open, adapts to any hand and cube
    closed = pinch < threshold
    n = len(pinch)

    grasp = next((i for i in range(n - HOLD_FRAMES) if closed[i:i + HOLD_FRAMES].all()), None)
    if grasp is None:
        raise RuntimeError("No grasp found in this video")

    release = next((i for i in range(grasp + HOLD_FRAMES, n - HOLD_FRAMES)
                    if not closed[i:i + HOLD_FRAMES].any()), None)
    if release is None:
        raise RuntimeError("Grasp found but no release. Does the video end before the cube is let go?")

    return grasp, release, threshold


def build_path(track_xy, start, end, n_points=PATH_POINTS):
    '''Takes the carried part of the track and resamples it to a fixed number of points, evenly spread in time.'''
    segment = np.asarray(track_xy)[start:end + 1]
    old = np.linspace(0, 1, len(segment))
    new = np.linspace(0, 1, n_points)
    return np.column_stack([np.interp(new, old, segment[:, 0]), np.interp(new, old, segment[:, 1])])


def process_video(video_path, debug=False):
    video_path = Path(video_path)

    H, _, _ = calibrate(str(video_path))  # finds the marker in this video's first frame

    debug_path = None
    if debug:
        Path("debug").mkdir(exist_ok=True)
        debug_path = Path("debug") / f"{video_path.stem}_tracked.mp4"

    points_px, pinches, fps = track_video(video_path, debug_path)
    hand_found = float((~np.isnan(pinches)).mean())

    pinch = smooth(fill_gaps(pinches))
    points_px = np.column_stack([fill_gaps(points_px[:, 0]), fill_gaps(points_px[:, 1])])
    track_xy = np.array([pixel_to_table(H, u, v) for u, v in points_px])

    grasp, release, threshold = find_grasp_and_release(pinch)

    # positions from a few steady frames rather than one, so tracking jitter averages out
    grasp_xy = np.median(track_xy[grasp:grasp + HOLD_FRAMES], axis=0)
    release_xy = np.median(track_xy[max(0, release - HOLD_FRAMES):release + 1], axis=0)

    demo = {
        "name": video_path.stem,
        "video": str(video_path),
        "fps": fps,
        "hand_found": round(hand_found, 3),
        "pinch_threshold": round(float(threshold), 3),
        "grasp_frame": int(grasp),
        "release_frame": int(release),
        "release_source": "vision",
        "grasp_xy": grasp_xy.round(4).tolist(),
        "release_xy": release_xy.round(4).tolist(),
        "path_xy": build_path(track_xy, grasp, release).round(4).tolist(),
        "track_xy": track_xy.round(4).tolist(),
    }

    Path("demos").mkdir(exist_ok=True)
    out_path = Path("demos") / f"{video_path.stem}.json"
    out_path.write_text(json.dumps(demo, indent=1))

    print(f"{video_path.stem}: hand found {hand_found:.0%}, grasp at {grasp / fps:.2f}s {demo['grasp_xy']}, "
          f"release at {release / fps:.2f}s {demo['release_xy']} -> {out_path}")
    return demo


if __name__ == "__main__":
    videos = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not videos:
        print(__doc__)
        sys.exit(1)

    for video in videos:
        try:
            process_video(video, debug="--debug" in sys.argv)
        except RuntimeError as error:
            print(f"{video}: {error}")