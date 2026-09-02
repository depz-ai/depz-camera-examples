# Zone intrusion detection with a depth camera

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/virtual-fence

Mark a zone on the floor and alarm on anything above it — no person detector, a top-down view, and the camera's 2 % bias measured a second, independent way.

![A guarded zone marked on the floor, with the alarm state and a top-down view beside the camera image](https://depz.ai/examples/cameras-06-fence.webp)

*The floor is tinted green. The zone — 1.20 × 0.86 m, set by two clicks — has a box standing in it, so everything turns red: the zone, the intruder, the frame. On the right is the plan: the camera at the bottom, a one-metre grid, the zone, and the intruder's footprint inside it.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** draw a rectangle on the floor with two mouse clicks, and have the camera raise the alarm when anything steps into it. Plus a bird's eye plan of the room, so you can see the zone from above.

## What you need

- Any DEPZ stereo camera on USB, mounted high enough to see the floor
- The same packages as the earlier projects — no hand tracking here

```bash
pip install -r requirements.txt
```

## Run it

```bash
python virtual_fence.py
```

Wait for the floor to turn green, then click two points on the floor: those are opposite corners of your zone. Walk into it, or put a box there.

- `c` — clear the zone and draw a new one, `s` — screenshot, `q` — quit

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Zone intrusion detection with a depth camera](https://depz.ai/developers/depth-cameras/example-projects/virtual-fence)**.

The complete program: [`virtual_fence.py`](./virtual_fence.py).
