# DEPZ Camera Examples — hands-on example projects for DEPZ stereo depth cameras

Seven small, self-contained example projects for the [DEPZ](https://depz.ai)
stereo depth cameras. Each one is a single Python file, runs in about ten
minutes on a desk, and ends in a number you can check with a tape measure.

Nothing here is a toy demo that only works in the video. Every example project was built
against real hardware and then argued with a ruler, and the measured error —
including where the camera reads short — is written down in each project’s full article on depz.ai.

**Web versions with interactive demos and the full write-ups:** https://depz.ai/developers/depth-cameras/example-projects

## The example projects

Each example project reuses a piece of the one before it, so reading them in order costs
you the least explaining.

| # | Example project | What it does | Checked against a tape |
|---|---|---|---|
| 1 | [First depth map](examples/01_first_depth_map/) | Connect the camera, show a depth map, print the distance under the cursor | camera 2.94 m, tape 2.99 m |
| 2 | [Virtual ruler](examples/02_virtual_ruler/) | Click two points, get the distance between them in metres | A4 sheet, short side: 0.297 m |
| 3 | [Where the floor is](examples/03_floor_detection/) | Point cloud, floor plane, camera height above it; floor green, obstacles red | camera 0.871 m, tape 0.875–0.877 m |
| 4 | [Box measure](examples/04_box_measure/) | Width × depth × height of a parcel, with a 3D wireframe drawn over the frame | 0.274 × 0.269 × 0.247 m vs a taped 0.27 × 0.27 × 0.25 |
| 5 | [Any table is a touchscreen](examples/05_touch_table/) | Table plane plus fingertip: a touch detected to the millimetre | glasses case 33.6 mm, ruler 35 mm |
| 6 | [Virtual fence](examples/06_virtual_fence/) | A guarded zone on the floor, an alarm, and a top-down view | ruler 78.3 cm, tape 80.0 cm |
| 7 | [Depth keying](examples/07_depth_keying/) | Cut the background by distance — a green screen without the green screen | cut at 1.20 m, vanishes at 1.25–1.30 m |

A "3D human skeleton" example project was planned and dropped: these lenses are too narrow
to fit a standing person into a room-sized frame, and bone lengths are not
something a tape measure can settle — so the example project could not keep the promise the
rest of the series makes.

## What you need

- A DEPZ stereo depth camera, connected over USB. It is recognised
  automatically, and its calibration is fetched from the cloud by serial number.
- Python 3.10 or newer.
- Windows 10+ (x64), Linux (x86_64 or ARM64, glibc 2.28+) or macOS 14+ on
  Apple Silicon. Intel Macs are not supported. The SDK itself also runs on
  Windows 11 ARM64, but these examples have not been tried there:
  `requirements.txt` asks for `opencv-python` (the examples draw their
  windows with it), and PyPI has no wheel of it for that platform.
- A tape measure. This is not a joke — it is half of every example project.

## Install

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` installs the SDK with OpenVINO (Intel / AMD CPUs, Intel
GPU and NPU) and the ONNX Runtime CPU engine. With an NVIDIA GPU, replace
`cpu` there by `cuda12-bundled` (and add `tensorrt12` for TensorRT); the
[install selector](https://depz.ai/developers/sdk) lists every option.

Then run any example project directly:

```bash
.venv/bin/python examples/01_first_depth_map/first_depth_map.py      # Windows: .venv\Scripts\python examples\01_first_depth_map\first_depth_map.py
```

Example project 5 needs MediaPipe, which has to be installed separately — see
[requirements.txt](requirements.txt) and the
[example project's README](examples/05_touch_table/) for why.

## Documentation

- SDK documentation and downloads: <https://depz.ai/developers>
- These examples on the website:
  <https://depz.ai/developers/depth-cameras/example-projects>
- The companion series for the DEPZ USB sensors:
  [depz-sensor-examples](https://github.com/depz-ai/depz-sensor-examples)

## Licence

[MIT](LICENSE) — use the code in your own projects, commercial ones included.
