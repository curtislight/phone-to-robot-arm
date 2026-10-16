'''Pixel to table calibration using an ArUco marker
'''

import sys
import cv2
import numpy as np
 
import config
 
DICTIONARY = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.MARKER_DICT))
 
 
def make_marker(path="marker.png", size_px=800):
    marker = cv2.aruco.generateImageMarker(DICTIONARY, config.MARKER_ID, size_px)
    # white border around the marker helps detection
    marker = cv2.copyMakeBorder(marker, 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, marker)
    print(f"Saved {path}. Print it, measure the black square, set MARKER_SIZE in config.py")
 
 
def load_image(path):
    '''Loads a photo, or the first frame of a video.'''
    if path.lower().endswith((".mp4", ".mov", ".avi")):
        video = cv2.VideoCapture(path)
        ok, frame = video.read()
        video.release()
        if not ok:
            raise RuntimeError(f"Could not read a frame from {path}")
        return frame
 
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Could not open {path}")
    return image
 
 
def find_marker_in(gray):
    detector = cv2.aruco.ArucoDetector(DICTIONARY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if marker_id == config.MARKER_ID:
            return marker_corners.reshape(4, 2)
    return None
 
 
def detect_marker(image):
    '''Returns the marker's 4 corners in pixels, ordered top-left, top-right,
    bottom-right, bottom-left (as printed). None if not found.'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = find_marker_in(gray)
    if corners is not None:
        return corners
 
    # Fallback for glare: a washed-out print can be as bright as the table.
    # The table is coloured and the marker isn't, so turn every colourful pixel white.
    # That leaves the grey border standing out against white.
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]
    for limit in (60, 50, 70):
        cleaned = gray.copy()
        cleaned[saturation > limit] = 255
        corners = find_marker_in(cleaned)
        if corners is not None:
            return corners
    return None
 
 
def marker_corners_on_table():
    '''The same 4 corners in table coordinates (metres), same order as OpenCV.
    Top of the marker points along +x, its left side along +y.'''
    cx, cy = config.MARKER_CENTER
    h = config.MARKER_SIZE / 2
    return np.array([
        [cx + h, cy + h],  # top-left
        [cx + h, cy - h],  # top-right
        [cx - h, cy - h],  # bottom-right
        [cx - h, cy + h],  # bottom-left
    ], dtype=np.float32)
 
 
def compute_homography(pixel_corners):
    '''The mapping from image pixels to table metres.'''
    H, _ = cv2.findHomography(pixel_corners.astype(np.float32), marker_corners_on_table())
    return H
 
 
def pixel_to_table(H, u, v):
    '''Converts one pixel (u, v) to a table position (x, y) in metres.'''
    point = np.array([[[u, v]]], dtype=np.float32)
    x, y = cv2.perspectiveTransform(point, H)[0, 0]
    return float(x), float(y)
 
 
def video_frames(path, step=5):
    '''Yields every step-th frame of a video.'''
    video = cv2.VideoCapture(path)
    index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if index % step == 0:
            yield frame
        index += 1
    video.release()
 
 
def calibrate(path):
    '''Detects the marker in a photo or video, returns the homography.
    For videos it tries frames until it finds a clear one: the first frame is
    often blurred from tapping record, and the camera doesn't move during a clip.'''
    if path.lower().endswith((".mp4", ".mov", ".avi")):
        frames = video_frames(path)
    else:
        frames = [load_image(path)]
 
    for image in frames:
        corners = detect_marker(image)
        if corners is not None:
            return compute_homography(corners), image, corners
 
    raise RuntimeError("Marker not found in any frame. Check lighting, focus, and that the whole marker is in shot")
 
 
def save_debug_image(image, corners, path="calib_debug.png"):
    '''Draws the detected corners so you can check the order is right.'''
    debug = image.copy()
    for (u, v), label in zip(corners, ["TL", "TR", "BR", "BL"]):
        cv2.circle(debug, (int(u), int(v)), 8, (0, 0, 255), -1)
        cv2.putText(debug, label, (int(u) + 10, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(path, debug)
    print(f"Saved {path}, check the corners are labelled in the right place")
 
 
def click_to_measure(image, H):
    '''Opens the image. Click anywhere on the table to print its table x, y.
    Press any key to close.'''
    scale = min(1.0, 1200 / max(image.shape[:2]))  # phone photos are too big for the screen
    shown = cv2.resize(image, None, fx=scale, fy=scale)
 
    def on_click(event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            x, y = pixel_to_table(H, u / scale, v / scale)  # undo the display scaling
            print(f"pixel ({u / scale:.0f}, {v / scale:.0f}) -> table x={x:.3f} m, y={y:.3f} m")
 
    cv2.imshow("click to measure", shown)
    cv2.setMouseCallback("click to measure", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
 
    if sys.argv[1] == "marker":
        make_marker()
        sys.exit(0)
 
    H, image, corners = calibrate(sys.argv[1])
    save_debug_image(image, corners)
    print("Marker corners (pixels):\n", corners.round(1))
 
    if len(sys.argv) > 2 and sys.argv[2] == "click":
        click_to_measure(image, H)