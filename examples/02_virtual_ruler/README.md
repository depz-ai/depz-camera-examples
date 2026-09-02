# Measure the distance between two points with a depth camera

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/virtual-ruler

Click two pixels, get metres. Why measuring across the view beats measuring into it — verified against an A4 sheet: 0.297 m.

![Two clicked points on the image with the distance between them printed in metres](https://depz.ai/examples/cameras-02-ruler.webp)

*Left: the rectified left eye, with the two picked points and the line between them. Right: the same two points on the depth map. The camera says the marks are 1.136 m apart.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** click two points anywhere in the picture and read the distance between them in metres — then argue with a tape measure.

> [The first depth map](https://depz.ai/developers/depth-cameras/example-projects/first-depth-map) answered how far is that. This project answers how big is that — and that is the difference between a curiosity and a tool.

## What you need

- Any DEPZ stereo camera (Varius / Minima / Clarity) on USB
- Python 3.11+ and the same packages as the first project:

```bash
pip install -r requirements.txt   # depz-camera-sdk, openvino, numpy, opencv-python
```

- A tape measure and something of a known size — a sheet of A4 paper is ideal, because you already know the answer to the millimetre (0.297 × 0.210 m).

## Run it

```bash
python virtual_ruler.py
```

The first run downloads a ~21 MB neural network. Click one point, click another, and the label shows the distance. A third click starts a fresh pair; `c` clears, `s` saves a screenshot, `q` quits. Either half of the window works — the two halves are the same pixels.

```python
def pick(x: int, y: int, w: int, h: int) -> None:
    """Remember a clicked pixel. A third click starts a fresh pair."""
    if len(picks) == 2:
        picks.clear()
        tracks.clear()
    picks.append((min(max(x, 0), w - 1), min(max(y, 0), h - 1)))
    tracks.append(deque(maxlen=SMOOTH_N))
```

Give each point a moment after clicking: the reading is the median of the last 3 frames, which at this model's ~2 fps takes about 1.4 seconds to fill.

```python
def follow(cloud: np.ndarray) -> None:
    """Add this frame's XYZ for every picked pixel."""
    for (x, y), hist in zip(picks, tracks):
        p = cloud[y, x]
        if np.isfinite(p).all() and p[2] > 0:
            hist.append(np.asarray(p, dtype=np.float64))


def steady(hist: deque) -> np.ndarray | None:
    """Median of the last readings — one frame on its own is too jumpy."""
    return np.median(np.stack(hist), axis=0) if hist else None
```

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Measure the distance between two points with a depth camera](https://depz.ai/developers/depth-cameras/example-projects/virtual-ruler)**.

The complete program: [`virtual_ruler.py`](./virtual_ruler.py).
