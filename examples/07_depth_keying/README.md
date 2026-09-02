# Remove the background by depth — a green screen without the green

> **Full article — the physics, the measured numbers, interactive demos:**
> https://depz.ai/developers/depth-cameras/example-projects/depth-keying

Cut the background by distance instead of colour: flying-pixel filtering, a feathered mask, and the measured 5–10 cm cost of a clean edge.

![A person in the foreground kept sharp while the background is blurred away by distance](https://depz.ai/examples/cameras-07-keying.webp)

*Everything closer than 1.60 m is kept, everything beyond it is blurred away — 29 % of the frame survived the cut. The mask is the model's, the anonymity is the author's.*

Hardware used: [DEPZ Minima stereo camera](https://depz.ai/product/stereo-camera-minima-wide-lenses) or [DEPZ Varius](https://depz.ai/product/stereo-camera-varius-regular-baseline-wide-lenses).

**Goal:** cut the background out by **distance**. No cloth, no lighting rig, no chroma spill — and it works in front of a bookshelf, a kitchen or a moving street, where a green screen would need a whole studio.

## What you need

- Any DEPZ stereo camera on USB
- The same packages as the earlier projects — no extra models here

```bash
pip install -r requirements.txt
python depth_keying.py
```

- `[` and `]` — move the cut nearer or further (it is printed in the header)
- `b` — background: blur → black → white
- `m` — show the raw mask, `s` — screenshot, `q` — quit

## The full write-up

This README is only the launch pad. The physics, the measured numbers, every flag and the interactive demos are in the full article: **[Remove the background by depth — a green screen without the green](https://depz.ai/developers/depth-cameras/example-projects/depth-keying)**.

The complete program: [`depth_keying.py`](./depth_keying.py).
