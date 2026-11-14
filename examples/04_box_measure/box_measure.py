"""Example project 4 — Measure a box.

Put a parcel on the floor, point the camera down at it, hover over it, and read
its width, depth and height in metres. Example projects 2 and 3 in one: the floor gives a
reference surface, the point cloud gives the size, nothing is clicked.
Keys: s = save a screenshot, q or Esc = quit.
"""
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline

MODEL = ("lite-any-stereo", "v2-s-640x384")   # the steady one, as in example projects 2-3
TRIES, SAMPLE = 600, 4000   # RANSAC: plane guesses per frame, points per vote
TOL = 0.02                  # a point this close to the plane counts as floor
MAX_TILT, MIN_DROP = 60, 0.30       # guards that keep walls and tables out
MIN_H, MAX_H = 0.03, 1.00   # a box stands between these heights above the floor
MIN_PIXELS = 400            # smaller lumps are noise, not parcels
SCALE = 2
SMOOTH_N = 5                # frames to median the printed size over (~2 s)

_recent: deque = deque(maxlen=SMOOTH_N)
_last_size = 0          # pixels in the previous lump, to spot a new one


def find_floor(pts: np.ndarray):
    """RANSAC: guess planes from random point triples, keep the most popular."""
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


def basis(normal: np.ndarray):
    """Two perpendicular directions that lie *in* the floor."""
    u = np.cross(normal, (1.0, 0.0, 0.0))
    if np.linalg.norm(u) < 1e-6:               # normal happened to face x
        u = np.cross(normal, (0.0, 0.0, 1.0))
    u = u / np.linalg.norm(u)
    return u, np.cross(normal, u)


def pick_object(heights, valid, cursor):
    """Split everything standing on the floor into lumps, take one of them."""
    mask = (valid & (heights > MIN_H) & (heights < MAX_H)).astype(np.uint8)
    found, tags = cv2.connectedComponents(mask)
    if found < 2:
        return None
    if cursor is None:
        return None
    chosen = int(tags[cursor[1], cursor[0]])   # the lump under the mouse, only
    if chosen == 0:                            # cursor is on bare floor
        return None
    sizes = np.bincount(tags.ravel())
    return tags == chosen if sizes[chosen] >= MIN_PIXELS else None


def top_face(hs: np.ndarray) -> float:
    """Height of the flat top: the busiest 5 mm slice of the lump.

    A parcel has one big horizontal top and thin vertical sides, so the top
    lands inside a single slice while the sides smear across dozens of them.
    Taking the highest point instead would just find the noisiest one — on a
    34 mm book that reads 45 mm, this reads 37 and holds it to 0.6 mm.
    """
    counts, edges = np.histogram(hs, bins=np.arange(MIN_H, hs.max() + 0.005, 0.005))
    peak = edges[int(counts.argmax())] + 0.0025
    return float(np.median(hs[np.abs(hs - peak) < 0.01]))


def measure(cloud, heights, sel, plane):
    """Drop the lid onto the floor as a shadow, box the shadow in."""
    normal, _ = plane
    u, v = basis(normal)
    top = top_face(heights[sel])
    lid = sel & (np.abs(heights - top) < 0.02)      # the lid alone
    pts = cloud[lid if lid.sum() > MIN_PIXELS else sel]
    shadow = np.stack([pts @ u, pts @ v], axis=1).astype(np.float32)
    middle = np.median(shadow, axis=0)         # a pixel on the silhouette sees
    reach = np.linalg.norm(shadow - middle, axis=1)   # the box and the wall at
    shadow = shadow[reach <= np.percentile(reach, 99)]   # once: throw those out
    rect = cv2.minAreaRect(shadow)             # orientation, at any angle
    ang = np.radians(rect[2])                  # then measure along its own axes
    axes = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
    local = (shadow - np.array(rect[0])) @ axes.T
    lo, hi = np.percentile(local, (0.2, 99.8), axis=0)   # a hair off each edge,
    side_a, side_b = hi - lo                             # never the raw min/max
    centre = np.array(rect[0]) + (lo + hi) / 2 @ axes
    rect = (tuple(centre), (float(side_a), float(side_b)), rect[2])
    return (max(side_a, side_b), min(side_a, side_b), top), rect


def wireframe(rect, height, plane, proj):
    """The eight corners of the parcel, as pixels."""
    normal, offset = plane
    u, v = basis(normal)
    origin = -offset * normal                  # a point lying on the floor
    corners = cv2.boxPoints(rect)
    out = []
    for lift in (0.0, height):
        for x, y in corners:
            p = proj @ np.append(origin + u * x + v * y - normal * lift, 1.0)
            out.append((int(round(p[0] / p[2])), int(round(p[1] / p[2]))))
    return out


def compose(pkt, plane, sel, size, rect, proj, cursor):
    """Left eye, the parcel tinted, its 3-D box drawn on top, sizes in words."""
    view = pkt.rectified_left.copy()
    if sel is not None:
        view[sel] = (0.45 * view[sel] + 0.55 * np.array((0, 200, 255))).astype(np.uint8)
    canvas = cv2.resize(view, (view.shape[1] * SCALE, view.shape[0] * SCALE),
                        interpolation=cv2.INTER_NEAREST)

    label = "point the camera down at a box on the floor"
    if plane is not None and size is None:
        label = "floor found - now hover over the box"
    if size is not None:
        global _last_size
        n = int(sel.sum())
        if abs(n - _last_size) > 0.3 * max(n, _last_size):   # a different lump
            _recent.clear()
        _last_size = n
        _recent.append(size)
        w, d, h = np.median(np.stack(_recent), axis=0)
        label = f"{w:.3f} x {d:.3f} x {h:.3f} m"
        pts = [(x * SCALE, y * SCALE) for x, y in wireframe(rect, h, plane, proj)]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0),      # floor rectangle
                     (4, 5), (5, 6), (6, 7), (7, 4),      # top rectangle
                     (0, 4), (1, 5), (2, 6), (3, 7)):     # the uprights
            cv2.line(canvas, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
    if cursor is not None:
        cv2.drawMarker(canvas, (cursor[0] * SCALE, cursor[1] * SCALE),
                       (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
        cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    cursor = None

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor
        cursor = (x // SCALE, y // SCALE)

    print("Starting the pipeline (the first run downloads a ~21 MB model)...")
    pipe = DepthPipeline(enable_rectified=True, enable_cloud=True,
                         inference=MODEL)
    pipe.start()
    proj = np.asarray(pipe.active_calibration.P_left_canvas, dtype=np.float64)
    cv2.namedWindow("box_measure")
    cv2.setMouseCallback("box_measure", on_mouse)
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:                  # camera died, don't hang
                print("no frames for 5 s — camera unplugged, or another"
                      " program is holding it?", pipe.last_error or "")
                break
            cloud = pkt.point_cloud
            valid = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
            pts = cloud[valid]
            plane = size = rect = sel = None
            if len(pts) > SAMPLE:
                plane = find_floor(pts[np.random.randint(0, len(pts), SAMPLE)])
            if plane is not None:
                heights = -(cloud @ plane[0] + plane[1])
                cur = None if cursor is None else (
                    min(max(cursor[0], 0), cloud.shape[1] - 1),
                    min(max(cursor[1], 0), cloud.shape[0] - 1))
                sel = pick_object(heights, valid, cur)
                if sel is not None:
                    size, rect = measure(cloud, heights, sel, plane)
            cv2.imshow("box_measure", compose(pkt, plane, sel, size, rect, proj, cursor))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, plane, sel, size, rect,
                                              proj, cursor))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
