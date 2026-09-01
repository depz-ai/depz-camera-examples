"""Example project 6 — A virtual fence on your floor.

Finds the floor (example project 3), finds whatever stands on it (example project 4), and raises an
alarm when something steps into a zone you drew. The right half is a bird's eye
plan of the room with a one-metre grid, so you can see the zone from above.

Click two points on the floor to set the zone corners.
Keys: c = clear the zone, s = screenshot, q or Esc = quit.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from depz_camera_sdk import DepthPipeline, set_cache_dir

# Models land in ./depz_sdk_cache relative to the *current* directory, so
# running from a different folder re-downloads them. Pin one shared cache.
set_cache_dir(Path.home() / ".cache" / "depz")

MODEL = ("lite-any-stereo", "v2-s-640x384")
TRIES, SAMPLE, TOL = 600, 4000, 0.02
MAX_TILT, MIN_DROP = 60, 0.30    # guards that keep walls and tables out
INTRUDER_H = 0.08                # taller than this above the floor = an intruder
                                 # (a 25 cm box counts, a doormat does not)
MAX_H = 2.20                     # ignore the ceiling
MIN_PIXELS = 250                 # fewer points than this in the zone = noise
EDGE_JUMP = 0.10                 # depth step (m) that marks a made-up edge pixel
PLAN_M, PLAN_PX = 5.0, 460       # bird's eye covers 5 m, drawn 460 px wide
SCALE = 2


def find_floor(pts: np.ndarray):
    """RANSAC plane fit — the same floor finder as example project 3."""
    tri = pts[np.random.randint(0, len(pts), (TRIES, 3))]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-6
    n, tri = n[keep] / ln[keep, None], tri[keep]
    n[n[:, 1] < 0] *= -1
    d = -(n * tri[:, 0]).sum(axis=1)
    votes = (np.abs(pts @ n.T + d) < TOL).sum(axis=0)
    tilt = np.degrees(np.arccos(np.clip(n[:, 1], -1, 1)))
    votes = np.where((tilt < MAX_TILT) & (-d > MIN_DROP), votes, 0)
    if votes.max() == 0:
        return None
    b = votes.argmax()
    inl = pts[np.abs(pts @ n[b] + d[b]) < TOL]
    centre = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centre, full_matrices=False)
    normal = vt[2] * (1 if vt[2][1] > 0 else -1)
    return normal, float(-(normal @ centre))


def basis(normal: np.ndarray):
    """Right and forward, as they lie *on* the floor.

    Example project 4 could take any two directions in the plane — a box is boxed in at
    whatever angle it sits. A map cannot: "forward" has to be the way the
    camera looks, or the plan comes out rotated. So take the camera's viewing
    axis, flatten it onto the floor, and cross it with the floor normal.
    """
    look = np.array([0.0, 0.0, 1.0])           # the camera looks along +Z
    forward = look - normal * float(look @ normal)
    if np.linalg.norm(forward) < 1e-6:         # camera stares straight down
        forward = np.array([0.0, 1.0, 0.0]) - normal * float(normal[1])
    forward = forward / np.linalg.norm(forward)
    return np.cross(normal, forward), forward  # (right, forward)


def solid(depth: np.ndarray) -> np.ndarray:
    """Drop pixels sitting on a depth cliff — they are invented, not measured.

    The network fills in *every* pixel, so along a silhouette it returns
    something halfway between the person and the wall behind them. Those points
    hang in mid-air and would trip the fence from outside. A real surface never
    jumps 10 cm within one pixel of itself; an edge always does.
    """
    k = np.ones((3, 3), np.uint8)
    high = cv2.dilate(depth, k)
    low = -cv2.dilate(-depth, k)
    return (high - low) < EDGE_JUMP


def steady_point(cloud, valid, x: int, y: int, win: int = 3):
    """The 3-D point under a click, medianed over its neighbours.

    One pixel is a noisy way to place a corner: at three metres a single pixel
    of disparity is worth centimetres (example project 1 measured exactly that). The median
    of a small patch of *floor* costs nothing and steadies the corner.
    """
    h, w = cloud.shape[:2]
    sel = valid[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
    patch = cloud[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1][sel]
    if len(patch) == 0:
        return None
    return patch[np.argsort(patch[:, 2])[len(patch) // 2]]


def to_floor(pts, plane):
    """Metres along the floor: where these points would stand if dropped down."""
    u, v = basis(plane[0])
    return np.stack([pts @ u, pts @ v], axis=1)


def inside(pt, zone) -> bool:
    (a0, b0), (a1, b1) = zone
    return min(a0, a1) <= pt[0] <= max(a0, a1) and min(b0, b1) <= pt[1] <= max(b0, b1)


def margin_m(pt, zone) -> float:
    """Metres from the point to the zone edge — negative means inside."""
    (a0, b0), (a1, b1) = zone
    dx = max(min(a0, a1) - pt[0], pt[0] - max(a0, a1))
    dy = max(min(b0, b1) - pt[1], pt[1] - max(b0, b1))
    if dx < 0 and dy < 0:
        return max(dx, dy)
    return float(np.hypot(max(dx, 0), max(dy, 0)))


def zone_pixels(zone, plane, proj):
    """The four corners of the zone, drawn where they really lie on the floor.

    Same projection example project 4 uses for the parcel's wireframe: build the corner in
    metres, push it through the camera matrix, divide by depth.
    """
    normal, offset = plane
    u, v = basis(normal)
    origin = -offset * normal                  # a point lying on the floor
    (a0, b0), (a1, b1) = zone
    out = []
    for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1)):
        p = proj @ np.append(origin + u * a + v * b, 1.0)
        if p[2] <= 0:
            return None                        # corner is behind the camera
        out.append((int(round(p[0] / p[2])), int(round(p[1] / p[2]))))
    return out


def plan_view(zone, marks, alarm: bool) -> np.ndarray:
    """Bird's eye: one-metre grid, the zone, the camera, and what is standing."""
    img = np.full((PLAN_PX, PLAN_PX, 3), 28, np.uint8)
    ppm = PLAN_PX / PLAN_M

    def xy(a, b):     # floor metres -> plan pixels, camera at the bottom centre
        return int(PLAN_PX / 2 + a * ppm), int(PLAN_PX - 20 - b * ppm)

    for m in range(-int(PLAN_M / 2), int(PLAN_M / 2) + 1):
        cv2.line(img, xy(m, -1), xy(m, PLAN_M), (55, 55, 55), 1)
        cv2.line(img, xy(-PLAN_M, m), xy(PLAN_M, m), (55, 55, 55), 1)
    cv2.putText(img, "1 m grid", (10, PLAN_PX - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(img, "camera", (PLAN_PX // 2 - 34, PLAN_PX - 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "^ away from the camera", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)

    if zone is not None:
        c = (0, 0, 255) if alarm else (0, 210, 0)
        cv2.rectangle(img, xy(*zone[0]), xy(*zone[1]), c, 2)
    for pt, hot, cloud_pts in marks:
        for a, b in cloud_pts:                 # the object's own footprint
            cv2.circle(img, xy(a, b), 1, (0, 0, 160) if hot else (90, 110, 120), -1)
        cv2.circle(img, xy(*pt), 7, (0, 0, 255) if hot else (0, 190, 255), -1)
    cv2.drawMarker(img, xy(0, 0), (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 16, 2)
    return img


def compose(pkt, plane, lumps, zone, marks, alarm, msg, proj, clicks) -> np.ndarray:
    """Camera view with the floor green and intruders red, plus the plan."""
    view = pkt.rectified_left.copy()
    h, w = pkt.depth.shape
    if plane is not None:
        heights = -(pkt.point_cloud @ plane[0] + plane[1])
        floor = np.abs(heights) < TOL
        view[floor] = (0.75 * view[floor] + 0.25 * np.array([0, 160, 0])).astype(np.uint8)
        floor_zone, intruder = lumps            # masks computed in main()
        if floor_zone is not None:              # the guarded patch of floor
            paint = np.array([0, 0, 230]) if alarm else np.array([0, 230, 90])
            view[floor_zone] = (0.45 * view[floor_zone] + 0.55 * paint).astype(np.uint8)
        if intruder is not None:                # whatever is standing in it
            view[intruder] = (0.35 * view[intruder]
                              + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    canvas = cv2.resize(view, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR)

    for x, y in clicks:                                   # corners picked so far
        cv2.drawMarker(canvas, (x * SCALE, y * SCALE), (255, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 22, 2)

    plan = plan_view(zone, marks, alarm)
    pad = np.full((h * SCALE, PLAN_PX, 3), 28, np.uint8)
    pad[:min(PLAN_PX, h * SCALE)] = plan[:min(PLAN_PX, h * SCALE)]
    canvas = cv2.hconcat([canvas, pad])

    if alarm:
        cv2.rectangle(canvas, (4, 4), (w * SCALE - 4, h * SCALE - 4), (0, 0, 255), 6)
    for i, text in enumerate([msg] if isinstance(msg, str) else msg):
        for colour, thick in (((0, 0, 0), 6), ((255, 255, 255), 2)):
            cv2.putText(canvas, text, (14, 42 + i * 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.95, colour, thick, cv2.LINE_AA)
    return canvas


def main() -> None:
    clicks, zone, frame_no = [], None, 0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x // SCALE, y // SCALE))

    pipe = DepthPipeline(enable_rectified=True, enable_cloud=True, inference=MODEL)
    pipe.start()
    proj = np.asarray(pipe.active_calibration.P_left_canvas, dtype=np.float64)
    cv2.namedWindow("virtual_fence")
    cv2.setMouseCallback("virtual_fence", on_mouse)
    try:
        while True:
            pkt = pipe.get(timeout=5.0)
            if pkt is None:
                print("no frames for 5 s — camera unplugged, or another program"
                      " is holding it?", pipe.last_error or "")
                break
            frame_no += 1
            cloud, (h, w) = pkt.point_cloud, pkt.depth.shape
            valid = (np.isfinite(cloud).all(axis=2) & (cloud[:, :, 2] > 0)
                     & solid(pkt.depth))
            pts = cloud[valid]
            plane = (find_floor(pts[np.random.randint(0, len(pts), min(SAMPLE, len(pts)))])
                     if len(pts) > 500 else None)

            lumps, marks, alarm = (None, None), [], False
            msg = ["no floor in view - point the camera at the ground"]
            if plane is not None:
                heights = -(cloud @ plane[0] + plane[1])
                if zone is None and len(clicks) >= 2:
                    picks = [steady_point(cloud, valid, *c) for c in clicks[-2:]]
                    if all(q is not None for q in picks):
                        corners = to_floor(np.stack(picks), plane)
                        zone = (tuple(corners[0]), tuple(corners[1]))
                        gap = float(np.linalg.norm(corners[0] - corners[1]))
                        print(f"zone set: {abs(corners[0][0]-corners[1][0]):.2f} x "
                              f"{abs(corners[0][1]-corners[1][1]):.2f} m | corners "
                              f"{gap:.3f} m apart (check this one with a tape)",
                              flush=True)
                    clicks.clear()

                floor = np.abs(heights) < TOL
                standing = valid & (heights > INTRUDER_H) & (heights < MAX_H)
                ground = to_floor(cloud.reshape(-1, 3), plane).reshape(h, w, 2)
                if zone is not None:
                    (a0, b0), (a1, b1) = zone
                    in_rect = ((ground[:, :, 0] >= min(a0, a1)) & (ground[:, :, 0] <= max(a0, a1))
                               & (ground[:, :, 1] >= min(b0, b1))
                               & (ground[:, :, 1] <= max(b0, b1)))
                    intruder = standing & in_rect      # anything in the air above the zone
                    intruder = cv2.erode(intruder.astype(np.uint8),
                                         np.ones((3, 3), np.uint8)).astype(bool)
                    alarm = int(intruder.sum()) >= MIN_PIXELS
                    lumps = (floor & in_rect, intruder if alarm else None)
                    marks = [(np.median(ground[intruder], axis=0), True,
                              ground[intruder][::25])] if alarm else []
                if zone is None:
                    marks = [(np.median(ground[standing], axis=0), False,
                              ground[standing][::40])] if standing.sum() > MIN_PIXELS else []
                if zone is None:
                    msg = ["click two points on the floor to set the zone"]
                else:
                    size = (abs(zone[0][0] - zone[1][0]), abs(zone[0][1] - zone[1][1]))
                    msg = [f"zone {size[0]:.2f} x {size[1]:.2f} m",
                           f"INTRUSION: {int(lumps[1].sum())} points inside" if alarm
                           else "clear"]

            cv2.imshow("virtual_fence", compose(pkt, plane, lumps, zone, marks, alarm, msg, proj, clicks))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                zone, clicks[:] = None, []
                print("zone cleared", flush=True)
            if key == ord("s"):
                out = Path(__file__).parent / f"screenshot_{int(time.time())}.png"
                cv2.imwrite(str(out), compose(pkt, plane, lumps, zone, marks, alarm,
                                              msg, proj, clicks))
                print("saved", out)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
