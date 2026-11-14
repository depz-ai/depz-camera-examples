# Get a depth map from a stereo camera in Python

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/first-depth-map

Connect a DEPZ stereo camera, read a live depth map, and learn what a depth pixel is worth — a real accuracy check: 2.94 m measured vs 2.99 m taped.

![The app window: the camera view beside its depth map, with the distance under the cursor](https://depz.ai/examples/cameras-01-depth-map.webp)

*Left: the camera's rectified left eye. Right: the depth map (warm colours = near, cold = far). The crosshair sits on a sheet of cardboard; the camera says 2.94 m, the tape measure says 2.99 m. Arguing with a tape measure is the whole point — [see below](#now-argue-with-a-tape-measure).*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** plug in a DEPZ stereo camera and, in under ten minutes, have a live window where every pixel knows how far away it is — with the distance in metres shown under your mouse cursor.

> The number under the cursor is the median of the last 9 frames, not a single reading. Why that matters is [explained below](#why-the-number-needs-nine-frames) — and it is the first real lesson about stereo depth.

## What you need

- Any DEPZ stereo camera (Varius / Minima / Clarity) on USB
- Python 3.10+ and three packages:

```bash
pip install -r requirements.txt   # depz-camera-sdk, openvino, numpy, opencv-python
```

No calibration step: the camera is recognised by its serial number and its factory calibration is fetched automatically on first use.

## Run it

```bash
python first_depth_map.py
```

The first run downloads a ~24 MB neural network; after that the pipeline starts in about 3 seconds. Move the mouse over either half of the window — the crosshair shows the same pixel in both, with its distance in metres. Press `s` to save a screenshot, `q` to quit.

```python
    if cursor is not None:
        cx, cy = cursor
        x = min(max(cx // SCALE, 0), 2 * w - 1) % w  # same pixel in either half
        y = min(max(cy // SCALE, 0), h - 1)          # cursor may leave the window
        z = depth_at(pkt.depth, x, y)
        label = f"{z:.2f} m" if np.isfinite(z) else "no data"
```

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Get a depth map from a stereo camera in Python](https://depz.ai/developers/depth-cameras/example-projects/first-depth-map)**.

The complete program: [`first_depth_map.py`](./first_depth_map.py).
