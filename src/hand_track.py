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
THRESHOLD_FRACTION = 0.65  # where the closed/open line sits, from most closed (0) to most open (1)
MIN_CARRY = 0.05     # release must be at least this far (metres) from the grab point
 
# yellow block detection, in OpenCV's HSV colour space (hue 0 to 179)
CUBE_HSV_LOW = (18, 80, 80)
CUBE_HSV_HIGH = (35, 255, 255)
CUBE_MIN_AREA = 500  # pixels, ignores small yellow specks
NEAR_CUBE = 0.05     # hand counts as "at the block" within this distance (metres)
 
 
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
    first_frame, last_frame = None, None
 
    with make_landmarker() as landmarker:
        frame_index = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if first_frame is None:
                first_frame = frame.copy()
            last_frame = frame.copy()
 
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
    return np.array(points), np.array(pinches), fps, first_frame, last_frame
 
 
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
 
 
def find_grasp_and_release(pinch, track_xy):
    '''Grasp: first time the pinch stays closed for HOLD_FRAMES.
    Release: right after the last frame the pinch is closed.'''
    # holding a cube keeps the fingers apart, so the line sits closer to open than to closed
    threshold = pinch.min() + THRESHOLD_FRACTION * (pinch.max() - pinch.min())
    closed = pinch < threshold
    n = len(pinch)
 
    grasp = next((i for i in range(n - HOLD_FRAMES) if closed[i:i + HOLD_FRAMES].all()), None)
    if grasp is None:
        raise RuntimeError("No grasp found in this video")
 
    # release: the last closed frame. After putting the cube down the hand opens and leaves,
    # so any flicker during the carry can't be mistaken for the release.
    release = int(np.nonzero(closed)[0][-1]) + 1
    release = min(release, n - 1)
    if np.linalg.norm(track_xy[release] - track_xy[grasp]) < MIN_CARRY:
        raise RuntimeError("Release is barely away from the grab point. Check the pinch graph in debug/")
 
    return grasp, release, threshold
 
 
def save_pinch_plot(pinch, threshold, grasp, release, fps, path):
    '''Graph of the pinch over time with the threshold and detected events, for tuning.'''
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window
    import matplotlib.pyplot as plt
 
    times = np.arange(len(pinch)) / fps
    plt.figure(figsize=(10, 4))
    plt.plot(times, pinch, label="pinch (thumb to index / hand size)")
    plt.axhline(threshold, color="grey", linestyle="--", label="threshold")
    plt.axvline(grasp / fps, color="green", label="grasp")
    plt.axvline(release / fps, color="red", label="release")
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
 
 
def find_cube_px(frame):
    '''Finds the yellow block by colour. Returns its centre in pixels and the mask, or None.'''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(CUBE_HSV_LOW), np.array(CUBE_HSV_HIGH))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # removes speckle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < CUBE_MIN_AREA:
        return None, mask
    m = cv2.moments(biggest)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]]), mask
 
 
def find_events_from_cube(track_xy, found, cube_start, cube_end):
    '''Grasp: first frame the hand is at the block's start.
    Release: last frame the hand is at the block's end.'''
    frames = np.nonzero(found)[0]  # only frames where the hand was actually seen
    near_start = frames[np.linalg.norm(track_xy[frames] - cube_start, axis=1) < NEAR_CUBE]
    near_end = frames[np.linalg.norm(track_xy[frames] - cube_end, axis=1) < NEAR_CUBE]
    if len(near_start) == 0 or len(near_end) == 0:
        raise RuntimeError("Hand never reached the block's start or end position")
    grasp, release = int(near_start[0]), int(near_end[-1])
    if release <= grasp:
        raise RuntimeError("Release came before grasp, check the block detection images in debug/")
    return grasp, release
 
 
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
 
    points_px, pinches, fps, first_frame, last_frame = track_video(video_path, debug_path)
    found = ~np.isnan(pinches)
    hand_found = float(found.mean())
 
    pinch = smooth(fill_gaps(pinches))
    points_px = np.column_stack([fill_gaps(points_px[:, 0]), fill_gaps(points_px[:, 1])])
    track_xy = np.array([pixel_to_table(H, u, v) for u, v in points_px])
 
    # first choice: the block's own start and end positions, seen with the hand out of shot
    start_px, start_mask = find_cube_px(first_frame)
    end_px, end_mask = find_cube_px(last_frame)
    if debug:
        cv2.imwrite(str(Path("debug") / f"{video_path.stem}_block_start.png"), start_mask)
        cv2.imwrite(str(Path("debug") / f"{video_path.stem}_block_end.png"), end_mask)
 
    threshold = pinch.min() + THRESHOLD_FRACTION * (pinch.max() - pinch.min())
    if start_px is not None and end_px is not None:
        grasp_xy = np.array(pixel_to_table(H, *start_px))
        release_xy = np.array(pixel_to_table(H, *end_px))
        grasp, release = find_events_from_cube(track_xy, found, grasp_xy, release_xy)
        event_source = "block"
    else:
        # fallback: the finger pinch
        grasp, release, threshold = find_grasp_and_release(pinch, track_xy)
        grasp_xy = np.median(track_xy[grasp:grasp + HOLD_FRAMES], axis=0)
        release_xy = np.median(track_xy[max(0, release - HOLD_FRAMES):release + 1], axis=0)
        event_source = "pinch"
 
    if debug:
        save_pinch_plot(pinch, threshold, grasp, release, fps, Path("debug") / f"{video_path.stem}_pinch.png")
 
    demo = {
        "name": video_path.stem,
        "video": str(video_path),
        "fps": fps,
        "hand_found": round(hand_found, 3),
        "pinch_threshold": round(float(threshold), 3),
        "grasp_frame": int(grasp),
        "release_frame": int(release),
        "release_source": "vision",
        "event_source": event_source,
        "grasp_xy": grasp_xy.round(4).tolist(),
        "release_xy": release_xy.round(4).tolist(),
        "path_xy": build_path(track_xy, grasp, release).round(4).tolist(),
        "track_xy": track_xy.round(4).tolist(),
    }
 
    Path("demos").mkdir(exist_ok=True)
    out_path = Path("demos") / f"{video_path.stem}.json"
    out_path.write_text(json.dumps(demo, indent=1))
 
    print(f"{video_path.stem} [{event_source}]: hand found {hand_found:.0%}, grasp at {grasp / fps:.2f}s {demo['grasp_xy']}, "
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