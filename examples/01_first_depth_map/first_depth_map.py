"""Example project 1 — Your first depth map.

Opens the first connected DEPZ camera, shows the rectified left eye next to
a colorized depth map, and prints the distance (in metres) under the mouse
cursor. Keys: s = save a screenshot, q or Esc = quit.
"""
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline

# The SDK's own default is the steadier lite-any-stereo. This example project
# runs the fast, jumpy lightstereo on purpose: its frame-to-frame wobble is
# what the nine-frame median below is there to tame. See the README.
MODEL = ("lightstereo", "s-sceneflow-opt-256x512")
MIN_M, MAX_M = 0.3, 6.0  # display range in metres — tweak for your scene
SCALE = 2                # on-screen upscale of the (model-sized) frames
SMOOTH_N = 9             # frames to median the cursor reading over (~0.3 s)

_recent: deque[float] = deque(maxlen=SMOOTH_N)
_last_px: tuple[int, int] | None = None


def depth_at(depth: np.ndarray, x: int, y: int) -> float:
    """Distance at one pixel, medianed over the last SMOOTH_N frames.

    One frame wobbles: disparity flickers by ~0.2 px, and since
    metres = f*B / disparity that flicker grows with range. Averaging over
    *pixels* would not help — neighbours flicker together; time does.
    """
    global _last_px
    if (x, y) != _last_px:      # cursor moved — don't mix two surfaces
        _recent.clear()
        _last_px = (x, y)
    z = float(depth[y, x])
    if np.isfinite(z) and z > 0:
        _recent.append(z)
    return float(np.median(_recent)) if _recent else float("nan")


def colorize(depth: np.ndarray) -> np.ndarray:
    """float32 metres -> BGR heatmap. Warm colours = near, cold = far."""
    valid = np.isfinite(depth) & (depth > 0)
    t = np.clip((depth - MIN_M) / (MAX_M - MIN_M), 0.0, 1.0)
    img = cv2.applyColorMap(((1.0 - t) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    return img


def compose(pkt, cursor: tuple[int, int] | None) -> np.ndarray:
    """Left view | depth map side by side, crosshair + metres at the cursor."""
    left = pkt.rectified_left          # already BGR — hand it to cv2 as is
    depth_img = colorize(pkt.depth)
    h, w = pkt.depth.shape
    canvas = cv2.hconcat([left, depth_img])
    canvas = cv2.resize(canvas, (2 * w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

    if cursor is not None:
        cx, cy = cursor
        x = min(max(cx // SCALE, 0), 2 * w - 1) % w  # same pixel in either half
        y = min(max(cy // SCALE, 0), h - 1)          # cursor may leave the window
        z = depth_at(pkt.depth, x, y)
        label = f"{z:.2f} m" if np.isfinite(z) else "no data"
        org = (x * SCALE + 12, y * SCALE - 10)
        for x0 in (x * SCALE, (x + w) * SCALE):  # crosshair on both halves
            cv2.drawMarker(canvas, (x0, y * SCALE), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 1)
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):  # outline
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    cursor = None

    def on_mouse(event, x, y, flags, param):
        nonlocal cursor
        cursor = (x, y)

    print("Starting the pipeline (the first run downloads a ~13 MB model)...")
    pipe = DepthPipeline(enable_rectified=True, inference=MODEL)
    pipe.start()
    cv2.namedWindow("first_depth_map")
    cv2.setMouseCallback("first_depth_map", on_mouse)
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:                  # camera died, don't hang
                print("no frames for 5 s — camera unplugged, or another"
                      " program is holding it?", pipe.last_error or "")
                break
            cv2.imshow("first_depth_map", compose(pkt, cursor))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, cursor))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
