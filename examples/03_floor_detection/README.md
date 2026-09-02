# Detect the floor plane in a point cloud with Python

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/floor-detection

Turn the depth map into a point cloud, find the floor by plane search, and measure the camera's height to 5 mm — a 34 mm book still reads as an obstacle.

![The scene with the floor painted green and everything above it red](https://depz.ai/examples/cameras-03-floor.webp)

*The floor is found, not assumed: nobody told the camera where the ground is or which way is down. The label says the camera is 0.877 m above it, tilted 26° down, and that the floor fills 29 % of the view. The crosshair sits on a book lying on the floor: 0.038 m up. The book is 34 mm thick by ruler, and the tape put the camera at 0.875–0.877 m.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** let the camera work out where the floor is, paint it green, paint everything standing on it red — and tell you how high the camera itself is above that floor. Which you can check with a tape measure in ten seconds.

> The first two projects read individual points. This one asks a question about the **whole scene at once**, and gets back something no single point can give you: a reference surface that every other height is measured against.

## What you need

- Any DEPZ stereo camera (Varius / Minima / Clarity) on USB
- The same packages as before:

```bash
pip install -r requirements.txt   # depz-camera-sdk, openvino, numpy, opencv-python
```

- A tape measure, and a camera pointed so that a decent patch of floor is in view. Anywhere from waist height looking down works well.

## Run it

```bash
python floor_detection.py
```

Green is the floor, red is anything at least 3 cm above it. The label shows the camera's height above the floor, how far down it is tilted, and how much of the view the floor takes. Move the mouse to read the height of any point above the floor. `s` saves a screenshot, `q` quits.

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Detect the floor plane in a point cloud with Python](https://depz.ai/developers/depth-cameras/example-projects/floor-detection)**.

The complete program: [`floor_detection.py`](./floor_detection.py).
