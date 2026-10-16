''' the release moment with sound

Putting the cube down makes a tap. Hand tracking only knows roughly when the
fingers opened. The tap pins down when the cube actually landed. This finds
sharp sounds (onsets) near the vision release and picks the loudest one,
then updates the demo's release frame, release position and carried path.
'''

import json
import subprocess
import sys
import tempfile
from pathlib import Path
 
import imageio_ffmpeg
import librosa
import numpy as np
 
from hand_track import HOLD_FRAMES, build_path
 
SEARCH_WINDOW = 0.5  # seconds either side of the vision release to look for the tap
 
 
def extract_audio(video_path, wav_path):
    '''Pulls the soundtrack out of the video as a mono WAV file using ffmpeg.'''
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path),
                    "-vn", "-ac", "1", str(wav_path)], check=True)
 
 
def find_tap(video_path, around_time):
    '''Returns the time (seconds) of the loudest sharp sound within SEARCH_WINDOW of around_time.
    None if there isn't one.'''
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio(video_path, wav_path)
        sound, sample_rate = librosa.load(wav_path, sr=None)
 
    strength = librosa.onset.onset_strength(y=sound, sr=sample_rate)  # how suddenly the sound changes, over time
    times = librosa.times_like(strength, sr=sample_rate)
    onsets = librosa.onset.onset_detect(onset_envelope=strength, sr=sample_rate)
 
    nearby = [o for o in onsets if abs(times[o] - around_time) <= SEARCH_WINDOW]
    if not nearby:
        return None
    loudest = max(nearby, key=lambda o: strength[o])
    return float(times[loudest])
 
 
def refine(video_path):
    video_path = Path(video_path)
    demo_path = Path("demos") / f"{video_path.stem}.json"
    if not demo_path.exists():
        print(f"{video_path.stem}: no demo file, run hand_track.py on it first")
        return None
    demo = json.loads(demo_path.read_text())
    fps = demo["fps"]
 
    # if this has been run before, start from the original vision estimate again
    vision_frame = demo.get("release_frame_vision", demo["release_frame"])
    vision_xy = demo.get("release_xy_vision", demo["release_xy"])
 
    tap_time = find_tap(video_path, vision_frame / fps)
    if tap_time is None:
        print(f"{video_path.stem}: no tap found near {vision_frame / fps:.2f}s, keeping the vision release")
        return demo
 
    track = np.array(demo["track_xy"])
    release = int(round(tap_time * fps))
    release = min(max(release, demo["grasp_frame"] + 1), len(track) - 1)  # keep it after the grasp and inside the video
 
    demo["release_frame_vision"] = vision_frame
    demo["release_xy_vision"] = vision_xy
    demo["release_frame"] = release
    demo["release_source"] = "audio"
    if demo.get("event_source") != "block":  # the block's own end position is more accurate than the hand's
        demo["release_xy"] = np.median(track[max(0, release - HOLD_FRAMES):release + 1], axis=0).round(4).tolist()
    demo["path_xy"] = build_path(track, demo["grasp_frame"], release).round(4).tolist()
 
    demo_path.write_text(json.dumps(demo, indent=1))
    shift_ms = (release - vision_frame) / fps * 1000
    print(f"{video_path.stem}: tap at {tap_time:.2f}s, release moved {shift_ms:+.0f}ms -> {demo['release_xy']}")
    return demo
 
 
if __name__ == "__main__":
    videos = sys.argv[1:]
    if not videos:
        print(__doc__)
        sys.exit(1)
 
    for video in videos:
        refine(video)