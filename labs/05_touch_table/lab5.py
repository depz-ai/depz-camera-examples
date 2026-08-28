relf crhbyijns crkflsdftn&
"""Lab 5 — Turn any table into a touchscreen.

Fits a plane to your table, tracks your fingertip, and registers a tap when the
finger comes within a few millimetres of the surface — anywhere on the table.
The height of the fingertip is printed in millimetres, so you can check it
against anything you can measure. Keys: c = calibrate (finger resting on the
table), r = reset the counter, s = screenshot, q or Esc = quit.
"""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

from depz_camera_sdk import DepthPipeline, set_cache_dir

# Models land in ./depz_sdk_cache relative to the *current* directory, so
# running from a different folder re-downloads them. Pin one shared cache.
set_cache_dir(Path.home() / ".cache" / "depz")

MODEL = ("lite-any-stereo", "v2-s-480x288")   # 3.5 fps, 1 mm of noise up close
HAND_TASK = Path.home() / ".cache" / "depz-labs" / "hand_landmarker.task"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker"
            "/hand_landmarker/float16/latest/hand_landmarker.task")
TIP = 8                # index fingertip in MediaPipe's 21-point hand
TRIES, SAMPLE = 600, 4000
TOL = 0.01             # a point within 1 cm of the plane counts as "table"
MAX_RANGE = 1.20       # further than this is the room, not the table
MAX_TILT = 60          # degrees, rejects walls
TOUCH_MM = 18.0        # fingertip closer than this to the table = a tap
PLANE_EVERY = 10       # the table does not move — refit the plane now and then
SCALE = 2

# No averaging over frames: this model's noise is already 1 mm, and a median
# would only add lag — the point of a tap is that it feels instant.


def find_plane(pts: np.ndarray):
    """RANSAC: guess planes from random point triples, keep the most popular.

    Same trick Lab 3 used for the floor — toss a sheet of cardboard at the
    cloud a few hundred times and keep the throw that touched the most points.
    """
    tri = pts[np.random.randint(0, len(pts), (TRIES, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-6
    n, tri = n[keep] / ln[keep, None], tri[keep]
    n[n[:, 1] < 0] *= -1                        # every normal points down
    d = -(n * tri[:, 0]).sum(axis=1)
    votes = (np.abs(pts @ n.T + d) < TOL).sum(axis=0)
    tilt = np.degrees(np.arccos(np.clip(n[:, 1], -1, 1)))
    votes = np.where(tilt < MAX_TILT, votes, 0)
    if votes.max() == 0:
        return None
    b = votes.argmax()                          # refine on every point it hit
    inl = pts[np.abs(pts @ n[b] + d[b]) < TOL]
    centre = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
    normal = vt[2] * (1 if vt[2][1] > 0 else -1)
    return normal, float(-(normal @ centre))


def fingertip(res, w: int, h: int):
    """Pixel of the index fingertip, clamped to the frame, or None."""
    if not res.hand_landmarks:
        return None
    k = res.hand_landmarks[0][TIP]
    return min(max(int(k.x * w), 0), w - 1), min(max(int(k.y * h), 0), h - 1)


def height_mm(cloud: np.ndarray, plane, px, base: float) -> float | None:
    """Millimetres from the table up to the fingertip.

    The camera sees the *top* of the pad of your finger, so a finger resting on
    the table reads about 10 mm, not zero. Press `c` to store that as `base`.
    """
    x, y = px
    patch = cloud[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3)
    good = patch[np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0)]
    if len(good) == 0:
        return None
    p = good[np.argsort(good[:, 2])[len(good) // 2]]
    return float(-(p @ plane[0] + plane[1])) * 1000 - base


def compose(pkt, plane, px, mm, touching: bool, taps: int) -> np.ndarray:
    """Left eye, table tinted green, fingertip marked, height and tap counter."""
    view = pkt.rectified_left.copy()
    h, w = pkt.depth.shape
    if plane is not None:
        on_table = np.abs(-(pkt.point_cloud @ plane[0] + plane[1])) < TOL
        view[on_table] = (0.65 * view[on_table] + 0.35 * np.array([0, 190, 0])).astype(np.uint8)
    canvas = cv2.resize(view, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR)

    if px is not None:
        centre = (px[0] * SCALE, px[1] * SCALE)
        if touching:                                  # a tap: bright ring plus a dot
            cv2.circle(canvas, centre, 26, (0, 255, 255), 3)
            cv2.circle(canvas, centre, 10, (0, 255, 255), -1)
        else:
            cv2.circle(canvas, centre, 10, (255, 0, 255), 2)

    lamp = (0, 255, 255) if touching else (70, 70, 70)
    cv2.circle(canvas, (w * SCALE - 60, 52), 26, lamp, -1)
    cv2.circle(canvas, (w * SCALE - 60, 52), 26, (255, 255, 255), 2)

    label = ("point the camera at a table" if plane is None else
             "show your hand" if mm is None else f"{mm:+.0f} mm above the table")
    for text, org, size in ((label, (14, 42), 1.0), (f"taps: {taps}", (14, 88), 0.9)):
        for colour, thick in (((0, 0, 0), 6), ((255, 255, 255), 2)):
            cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick,
                        cv2.LINE_AA)
    return canvas


def main() -> None:
    if not HAND_TASK.exists():          # 7.8 MB, fetched once, like the SDK does
        HAND_TASK.parent.mkdir(parents=True, exist_ok=True)
        print("downloading the hand model (7.8 MB, first run only)...")
        urllib.request.urlretrieve(HAND_URL, HAND_TASK)
    detector = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1))
    pipe = DepthPipeline(enable_rectified=True, enable_cloud=True, inference=MODEL)
    pipe.start()
    cv2.namedWindow("lab5")
    base, taps, was_down, frame_no, plane = 0.0, 0, False, 0, None
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:
                print("no frames for 5 s — camera unplugged, or another program"
                      " is holding it?", pipe.last_error or "")
                break
            frame_no += 1
            h, w = pkt.depth.shape
            res = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(pkt.rectified_left, cv2.COLOR_BGR2RGB)),
                frame_no * 100)
            px = fingertip(res, w, h)

            if frame_no % PLANE_EVERY == 1 or plane is None:
                keep = np.ones((h, w), bool)      # your hand is not part of the table
                if res.hand_landmarks:
                    xs = [min(max(int(k.x * w), 0), w - 1) for k in res.hand_landmarks[0]]
                    ys = [min(max(int(k.y * h), 0), h - 1) for k in res.hand_landmarks[0]]
                    keep[max(0, min(ys) - 30):max(ys) + 30,
                         max(0, min(xs) - 30):max(xs) + 30] = False
                flat = pkt.point_cloud.reshape(-1, 3)
                usable = (np.isfinite(flat).all(axis=1) & (flat[:, 2] > 0)
                          & (flat[:, 2] < MAX_RANGE) & keep.reshape(-1))
                pts = flat[usable]
                if len(pts) > 500:
                    plane = find_plane(pts[np.random.randint(
                        0, len(pts), min(SAMPLE, len(pts)))]) or plane

            mm = height_mm(pkt.point_cloud, plane, px, base) if (plane and px) else None
            touching = mm is not None and mm < TOUCH_MM
            if touching and not was_down:                 # count the press, not the hold
                taps += 1
                print(f"tap {taps} at {mm:+.0f} mm")
            was_down = touching

            cv2.imshow("lab5", compose(pkt, plane, px, mm, touching, taps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and mm is not None:
                base += mm                       # the finger is resting on the table now
                print(f"calibrated: a resting finger now reads 0 (offset {base:+.0f} mm)")
            if key == ord("r"):
                taps = 0
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, plane, px, mm, touching, taps))
                print("saved", out)
    finally:
        pipe.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
