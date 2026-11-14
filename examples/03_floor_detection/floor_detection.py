"""Example project 3 — Find the floor.

Fits a plane to the point cloud, paints the floor green and everything
standing on it red, and prints how high the camera is above that floor.
Move the mouse to read the height of any point above the floor.
Keys: s = save a screenshot, q or Esc = quit.
"""
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline

# The small lightstereo model is faster but its floor plane wobbles by 24 mm from frame
# to frame; this one wobbles by 8 mm and lands closer to the tape measure
# (870 mm vs a taped 876). A height you can check with a ruler is the whole
# point of this example project, so it pays 5 -> 2.3 fps for it. See the README.
MODEL = ("lite-any-stereo", "v2-s-640x384")

TRIES = 600        # random plane guesses per frame
SAMPLE = 4000      # points drawn from the cloud for the search
TOL = 0.02         # a point within 2 cm of the plane counts as "on the floor"
STEP_UP = 0.03     # anything this far above the floor is an obstacle
MAX_TILT = 60      # degrees: rejects walls, still lets you tilt the camera
MIN_DROP = 0.30    # the floor has to be at least this far below the camera
SCALE = 2
SMOOTH_N = 5       # frames to median the printed camera height over (~2 s)

_recent: deque[float] = deque(maxlen=SMOOTH_N)


def find_floor(pts: np.ndarray) -> tuple[np.ndarray, float] | None:
    """RANSAC: guess planes from random point triples, keep the most popular.

    Think of tossing a sheet of cardboard at the cloud a few hundred times and
    keeping the throw that touched the most points. Returns the plane as a
    normal vector and an offset, or None if no floor-like plane was found.
    """
    tri = pts[np.random.randint(0, len(pts), (TRIES, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(n, axis=1)
    keep = length > 1e-6
    n, tri = n[keep] / length[keep, None], tri[keep]
    n[n[:, 1] < 0] *= -1                       # make every normal point down
    d = -(n * tri[:, 0]).sum(axis=1)

    votes = (np.abs(pts @ n.T + d) < TOL).sum(axis=0)
    tilt = np.degrees(np.arccos(np.clip(n[:, 1], -1, 1)))
    votes = np.where((tilt < MAX_TILT) & (-d > MIN_DROP), votes, 0)
    if votes.max() == 0:
        return None

    best = votes.argmax()                      # refine on every point it hit
    inliers = pts[np.abs(pts @ n[best] + d[best]) < TOL]
    centre = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centre, full_matrices=False)
    normal = vt[2] * (1 if vt[2][1] > 0 else -1)
    return normal, float(-(normal @ centre))


def heights_above(cloud: np.ndarray, plane) -> np.ndarray:
    """Metres above the floor for every pixel (negative = below it)."""
    normal, offset = plane
    return -(cloud @ normal + offset)


def compose(pkt, plane, cursor: tuple[int, int] | None) -> np.ndarray:
    """Left eye with the floor painted green and obstacles red."""
    view = pkt.rectified_left.copy()
    label = "no floor in view - point the camera at the ground"
    h = None if plane is None else heights_above(pkt.point_cloud, plane)

    if h is not None:
        valid = np.isfinite(h) & (pkt.depth > 0)
        floor = valid & (np.abs(h) < TOL)
        obstacle = valid & (h > STEP_UP)
        for mask, colour in ((floor, (0, 220, 0)), (obstacle, (0, 0, 230))):
            view[mask] = (0.5 * view[mask] + 0.5 * np.array(colour)).astype(np.uint8)

        _recent.append(-plane[1])              # camera height above the plane
        tilt = np.degrees(np.arccos(np.clip(plane[0][1], -1, 1)))
        label = (f"camera {np.median(_recent):.3f} m above the floor, "
                 f"{tilt:.0f} deg down | floor {floor.mean() * 100:.0f}%")

    canvas = cv2.resize(view, (view.shape[1] * SCALE, view.shape[0] * SCALE),
                        interpolation=cv2.INTER_NEAREST)
    if cursor is not None and h is not None:
        x = min(max(cursor[0] // SCALE, 0), h.shape[1] - 1)
        y = min(max(cursor[1] // SCALE, 0), h.shape[0] - 1)
        cv2.drawMarker(canvas, (x * SCALE, y * SCALE), (255, 255, 255),
                       cv2.MARKER_CROSS, 20, 1)
        if np.isfinite(h[y, x]):
            label += f" | cursor {h[y, x]:+.3f} m"
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    cursor = None

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor
        cursor = (x, y)

    print("Starting the pipeline (the first run downloads a ~21 MB model)...")
    pipe = DepthPipeline(enable_rectified=True, enable_cloud=True,
                         inference=MODEL)
    pipe.start()
    cv2.namedWindow("floor_detection")
    cv2.setMouseCallback("floor_detection", on_mouse)
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:                  # camera died, don't hang
                print("no frames for 5 s — camera unplugged, or another"
                      " program is holding it?", pipe.last_error or "")
                break
            cloud = pkt.point_cloud
            good = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
            pts = cloud[good]
            plane = None
            if len(pts) > SAMPLE:
                plane = find_floor(pts[np.random.randint(0, len(pts), SAMPLE)])
            cv2.imshow("floor_detection", compose(pkt, plane, cursor))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, plane, cursor))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
