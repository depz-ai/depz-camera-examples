"""Example project 2 — A virtual ruler.

Click two points in the picture and read the distance between them in
centimetres. Every pixel of the point cloud carries X, Y, Z in metres, so the
answer is Pythagoras in three dimensions — no trigonometry of your own.
Keys: c = clear the pair, s = save a screenshot, q or Esc = quit.
"""
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline

# The lightstereo of example project 1 is faster but jumpier: on a 25 cm target
# its reading wobbles by ~4 cm from frame to frame, this one by ~0.3 cm. A ruler needs
# the steady one, and pays 6 -> 2.5 fps for it. See the README.
MODEL = ("lite-any-stereo", "v2-s-640x384")
MIN_M, MAX_M = 0.3, 6.0   # depth colouring range in metres
SMOOTH_N = 3              # frames to median each picked point over (~1.4 s)

picks: list[tuple[int, int]] = []   # pixels you clicked
tracks: list[deque] = []            # recent XYZ readings, one deque per pick


def pick(x: int, y: int, w: int, h: int) -> None:
    """Remember a clicked pixel. A third click starts a fresh pair."""
    if len(picks) == 2:
        picks.clear()
        tracks.clear()
    picks.append((min(max(x, 0), w - 1), min(max(y, 0), h - 1)))
    tracks.append(deque(maxlen=SMOOTH_N))


def follow(cloud: np.ndarray) -> None:
    """Add this frame's XYZ for every picked pixel."""
    for (x, y), hist in zip(picks, tracks):
        p = cloud[y, x]
        if np.isfinite(p).all() and p[2] > 0:
            hist.append(np.asarray(p, dtype=np.float64))


def steady(hist: deque) -> np.ndarray | None:
    """Median of the last readings — one frame on its own is too jumpy."""
    return np.median(np.stack(hist), axis=0) if hist else None


def colorize(depth: np.ndarray) -> np.ndarray:
    """float32 metres -> BGR heatmap. Warm colours = near, cold = far."""
    valid = np.isfinite(depth) & (depth > 0)
    t = np.clip((depth - MIN_M) / (MAX_M - MIN_M), 0.0, 1.0)
    img = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def compose(pkt) -> np.ndarray:
    """Left view | depth map, with the picked points and the measurement."""
    h, w = pkt.depth.shape
    canvas = cv2.hconcat([pkt.rectified_left, colorize(pkt.depth)])
    pts = [steady(t) for t in tracks]

    for x, y in picks:
        for x0 in (x, x + w):        # the same point on both halves
            cv2.drawMarker(canvas, (x0, y), (255, 255, 255),
                           cv2.MARKER_CROSS, 18, 2)
    if len(picks) == 2:
        for x0 in (0, w):
            cv2.line(canvas, (picks[0][0] + x0, picks[0][1]),
                     (picks[1][0] + x0, picks[1][1]), (255, 255, 255), 1)

    label = "click two points"
    if len(pts) == 2 and pts[0] is not None and pts[1] is not None:
        label = f"{float(np.linalg.norm(pts[0] - pts[1])):.3f} m"
    for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):   # outline
        cv2.putText(canvas, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    print("Starting the pipeline (the first run downloads a ~21 MB model)...")
    pipe = DepthPipeline(enable_rectified=True, enable_cloud=True,
                         inference=MODEL)
    pipe.start()
    first = pipe.get(timeout=20.0)          # the model warms up on frame one
    if first is None:
        raise SystemExit(f"no frames from the camera: {pipe.last_error or ''}")
    h, w = first.depth.shape

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pick(x % w, y, w, h)     # click either half, same pixel either way

    cv2.namedWindow("virtual_ruler")
    cv2.setMouseCallback("virtual_ruler", on_mouse)
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:                  # camera died, don't hang
                print("no frames for 5 s — camera unplugged, or another"
                      " program is holding it?", pipe.last_error or "")
                break
            follow(pkt.point_cloud)
            cv2.imshow("virtual_ruler", compose(pkt))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                picks.clear()
                tracks.clear()
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
