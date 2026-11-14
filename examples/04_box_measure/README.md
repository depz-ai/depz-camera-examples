# Measure box dimensions with a depth camera in Python

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/box-measure

Width, depth and height of a parcel — no markers, no clicking corners. Verified: a 27 × 27 × 25 cm box measured to within 3 mm.

![A cardboard box with a 3D wireframe drawn over it and its dimensions in the corner](https://depz.ai/examples/cameras-04-box.webp)

*The label reads 0.274 × 0.269 × 0.247 m. The tape says 0.27 × 0.27 × 0.25. The wireframe is not a drawing over the picture — its eight corners are real points in space, projected back into the frame.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** put a parcel on the floor, hover the mouse over it, and read its width, depth and height. No clicking, no markers, no ruler — and then check every one of the three numbers with a ruler.

> This is the [virtual ruler](https://depz.ai/developers/depth-cameras/example-projects/virtual-ruler) and [floor detection](https://depz.ai/developers/depth-cameras/example-projects/floor-detection) collected in one place. Floor detection found the floor; that floor is what makes "height" mean anything. The ruler measured between two points; here the points come from the shape itself instead of from your mouse.

## What you need

- Any DEPZ stereo camera (Varius / Minima / Clarity) on USB
- The same packages as before:

```bash
pip install -r requirements.txt   # depz-camera-sdk with its engines, numpy, opencv-python
```

- A cardboard box and a tape measure.
- **The camera above the box, looking down.** This one matters: the width and depth are read off the lid, so the lid has to be visible. From floor level you see two sides and no top, and the far edge of the parcel is anybody's guess.
- **Space around the box.** The project grabs one connected lump of stuff standing on the floor. A parcel touching the sofa is, as far as the geometry is concerned, part of the sofa.

## Run it

```bash
python box_measure.py
```

Point the camera down until the floor turns up, move the mouse onto the box, wait a second for the numbers to settle. `s` saves a screenshot, `q` quits.

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Measure box dimensions with a depth camera in Python](https://depz.ai/developers/depth-cameras/example-projects/box-measure)**.

The complete program: [`box_measure.py`](./box_measure.py).
