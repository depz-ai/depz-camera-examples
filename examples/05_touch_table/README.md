# Turn any table into a touchscreen with a depth camera

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/touch-table

Detect fingertip touches on an ordinary table: a plane fit that ignores your hand, no temporal smoothing, and 1.4 mm accuracy at an 18 mm threshold.

![A hand touching a table, with the touch indicator lit and a tap counter](https://depz.ai/examples/cameras-05-touch.webp)

*The table is tinted green — that is the plane the camera fitted to it. The yellow ring is the fingertip, the lamp in the corner lights up on contact, and ten taps have been logged. The header reads −3 mm: the project has been calibrated, so a touching finger sits at zero give or take a couple of millimetres.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** point the camera at an ordinary table, and have it tell you — to the millimetre — how far your fingertip is above the surface. Close enough, and that is a tap.

## What you need

- Any DEPZ stereo camera on USB, looking down at a table from ~30 cm above it
- Python 3.11+ and the usual packages, plus MediaPipe for hand tracking:

```bash
pip install -r requirements.txt
pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice certifi matplotlib
```

MediaPipe needs the `--no-deps` line: left to itself it pulls in a second OpenCV package (`opencv-contrib-python`, a major version ahead), which silently replaces the one the earlier projects are tested against. The three packages on the last line are what it actually needs to import.

The hand model (7.8 MB) downloads itself on the first run, the way the SDK fetches its own models.

## Run it

```bash
python touch_table.py
```

Show your hand. The header prints the height of your index fingertip above the table; touch the surface anywhere and the lamp lights up.

- `c` — calibrate: rest your finger on the table and press it
- `r` — reset the counter, `s` — screenshot, `q` — quit

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Turn any table into a touchscreen with a depth camera](https://depz.ai/developers/depth-cameras/example-projects/touch-table)**.

The complete program: [`touch_table.py`](./touch_table.py).
