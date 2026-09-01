"""Example project 7 — A green screen without the green screen.

Keeps everything closer than a distance you choose and replaces the rest.
No cloth, no lighting rig, no chroma spill: the cut is made on depth, so it
works against a bookshelf, a kitchen, or a moving street.

Keys: [ and ] move the cut nearer/further, b = background (blur / black /
white), m = show the raw mask, s = screenshot, q or Esc = quit.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline, set_cache_dir

# Models land in ./depz_sdk_cache relative to the *current* directory, so
# running from a different folder re-downloads them. Pin one shared cache.
set_cache_dir(Path.home() / ".cache" / "depz")

MODEL = ("lightstereo", "s-sceneflow-opt-256x512")  # fast; keying wants frames, not millimetres
CUT_M = 1.60           # start here; [ and ] move it
STEP_M = 0.10
EDGE_JUMP = 0.10       # depth step (m) that marks an invented edge pixel
FEATHER = 9            # softens the cut so hair does not look cut out with scissors
SCALE = 2


def solid(depth: np.ndarray) -> np.ndarray:
    """Drop pixels sitting on a depth cliff — they are invented, not measured.

    The network fills in every pixel, so along your outline it returns
    something halfway between you and the wall. Left alone, those points glow
    as a halo around your head.
    """
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(depth, k) + cv2.dilate(-depth, k)) < EDGE_JUMP


def cutout(depth: np.ndarray, cut: float) -> np.ndarray:
    """Soft 0..1 mask: 1 where the pixel is nearer than the cut."""
    near = (np.isfinite(depth) & (depth > 0) & (depth < cut) & solid(depth))
    m = near.astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)      # fill pinholes in the subject
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)       # drop specks in the background
    m = cv2.GaussianBlur(m, (FEATHER, FEATHER), 0)   # feather the edge
    return (m.astype(np.float32) / 255.0)[:, :, None]


def compose(pkt, cut: float, mode: int, show_mask: bool) -> np.ndarray:
    """The subject over a new background, or the raw mask if you asked for it."""
    frame = pkt.rectified_left
    alpha = cutout(pkt.depth, cut)
    if show_mask:
        out = (alpha * 255).astype(np.uint8).repeat(3, axis=2)
    else:
        back = (cv2.GaussianBlur(frame, (0, 0), 12) if mode == 0 else
                np.zeros_like(frame) if mode == 1 else np.full_like(frame, 235))
        out = (frame * alpha + back * (1 - alpha)).astype(np.uint8)

    h, w = pkt.depth.shape
    canvas = cv2.resize(out, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR)
    kept = float(alpha.mean()) * 100
    for i, text in enumerate([f"keeping everything closer than {cut:.2f} m",
                              f"{kept:.0f}% of the frame kept   [ ] to move the cut"]):
        for colour, thick in (((0, 0, 0), 6), ((255, 255, 255), 2)):
            cv2.putText(canvas, text, (14, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    pipe = DepthPipeline(enable_rectified=True, inference=MODEL)
    pipe.start()
    cv2.namedWindow("depth_keying")
    cut, mode, show_mask = CUT_M, 0, False
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:
                print("no frames for 5 s — camera unplugged, or another program"
                      " is holding it?", pipe.last_error or "")
                break
            cv2.imshow("depth_keying", compose(pkt, cut, mode, show_mask))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                cut = max(0.30, cut - STEP_M)
                print(f"cut at {cut:.2f} m")
            if key == ord("]"):
                cut += STEP_M
                print(f"cut at {cut:.2f} m")
            if key == ord("b"):
                mode = (mode + 1) % 3
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, cut, mode, show_mask))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
