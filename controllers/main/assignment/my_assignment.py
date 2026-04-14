import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_PROJECT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_IMG_DIR       = os.path.join(_PROJECT_DIR, 'gate_detections')
_TRUTH_FILE    = os.path.join(_PROJECT_DIR, 'gate_truth.json')
_RESULTS_FILE  = os.path.join(_PROJECT_DIR, 'gate_positions.txt')
os.makedirs(_IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera intrinsics  (Webots ideal pinhole, no distortion)
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 300, 300
FOV  = 1.5
F_PX = (IMG_W / 2) / np.tan(FOV / 2)   # ≈ 161 px

_K    = np.array([[F_PX, 0,    IMG_W / 2],
                  [0,    F_PX, IMG_H / 2],
                  [0,    0,    1         ]], dtype=np.float64)
_DIST = np.zeros((4, 1), dtype=np.float64)

# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------
GATE_PANEL_H = 0.40   # m – square pink panel side length
_HALF        = GATE_PANEL_H / 2

# 3D corners of the square panel in its own frame (centred at origin, Z=0).
# Order: top-left, top-right, bottom-right, bottom-left  (clockwise from TL)
# required by SOLVEPNP_IPPE_SQUARE: (-h/2, h/2), (h/2, h/2), (h/2, -h/2), (-h/2, -h/2)
_OBJ_PTS = np.array([[-_HALF,  _HALF, 0],
                     [ _HALF,  _HALF, 0],
                     [ _HALF, -_HALF, 0],
                     [-_HALF, -_HALF, 0]], dtype=np.float32)

# ---------------------------------------------------------------------------
# Detection – HSV thresholds for pink/magenta (hue wraps around 0 / 179)
# ---------------------------------------------------------------------------
HSV_LO1 = np.array([140,  60,  40])
HSV_HI1 = np.array([179, 255, 255])
HSV_LO2 = np.array([  0,  60,  40])
HSV_HI2 = np.array([ 10, 255, 255])
BORDER       = 8
MIN_AREA     = 200
MIN_SOLIDITY = 0.55
MAX_ASPECT   = 2.5
MIN_PIXEL_H  = 6

# Minimum XY distance (m) between saved gate positions – prevents saving the
# same gate twice as the drone moves past it.
_SAVE_DEDUP_RADIUS  = 1.5   # m – gates are ≥2 m apart; increased to avoid re-saving
_FULL_PANEL_MARGIN  = 20   # px – reject if bounding box touches frame edge (partial panels give bad PnP)
NUM_GATES          = 5

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
CRUISE_Z         = 1.0   # m – default flight altitude
APPROACH_DIST    = 1.5   # m – approach waypoint in front of gate centre
PAST_DIST        = 1.2   # m – how far past the gate centre before switching to SEARCH
TRAJ_STEP_THRESH = 0.15  # m – advance to next trajectory setpoint when within this distance
T_SEG            = 2.5   # s – time budget per waypoint-to-waypoint segment
DISC_STEPS       = 20    # trajectory points per segment
SEARCH_YAW_RATE  = 0.04  # rad per control step while rotating to find next gate

# Arena bounds for trajectory clipping
ARENA_XMIN, ARENA_XMAX = 0.0, 9.0
ARENA_YMIN, ARENA_YMAX = 0.0, 9.0



# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_gate_corners(bgr_image):
    """Return a list of (4,2) float32 corner arrays, one per detected panel."""
    hsv  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, HSV_LO1, HSV_HI1),
                          cv2.inRange(hsv, HSV_LO2, HSV_HI2))
    mask[:BORDER,  :]  = 0
    mask[-BORDER:, :]  = 0
    mask[:,  :BORDER]  = 0
    mask[:, -BORDER:]  = 0
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or (area / hull_area) < MIN_SOLIDITY:
            continue
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if min(rw, rh) < 1 or max(rw, rh) / (min(rw, rh) + 1e-6) > MAX_ASPECT:
            continue
        results.append(cv2.boxPoints(rect).astype(np.float32))
    return results


# ---------------------------------------------------------------------------
# PnP pose estimation
# ---------------------------------------------------------------------------
def _sort_corners_tl_tr_br_bl(corners):
    """
    Sort 4 pixel corners into [top-left, top-right, bottom-right, bottom-left].
    Required ordering for SOLVEPNP_IPPE_SQUARE.
    """
    by_y   = corners[np.argsort(corners[:, 1])]   # smallest y = top
    top    = by_y[:2][np.argsort(by_y[:2, 0])]    # left then right among top two
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]    # left then right among bottom two
    # tl, tr, br, bl
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def pnp_gate_pose(corners, sensor_data):
    """
    Estimate the gate panel centre in world coordinates using PnP.

    IPPE_SQUARE returns two mirror-image solutions.  For distant panels the
    reprojection errors of both solutions are nearly identical, so argmin is
    unreliable.  Instead we use the independent depth-from-height formula
    (F_PX * GATE_PANEL_H / pixel_h) as a reference and pick the IPPE solution
    whose forward distance (tvec[2] in camera frame) is closest to it.

    Returns a (3,) world-frame position array, or None on failure.
    """
    sorted_c = _sort_corners_tl_tr_br_bl(corners)
    img_pts  = sorted_c.reshape(-1, 1, 2).astype(np.float64)

    # Reference depth from known physical panel height – independent of PnP.
    # Average the left and right vertical edge lengths for robustness.
    pixel_h_l = float(np.linalg.norm(sorted_c[0] - sorted_c[3]))  # TL→BL
    pixel_h_r = float(np.linalg.norm(sorted_c[1] - sorted_c[2]))  # TR→BR
    pixel_h   = (pixel_h_l + pixel_h_r) / 2.0
    if pixel_h < 1.0:
        return None
    depth_ref = F_PX * GATE_PANEL_H / pixel_h   # forward distance reference

    try:
        retval, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            _OBJ_PTS, img_pts, _K, _DIST,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None

    if not retval:
        return None

    # Select the physically correct solution geometrically:
    #   1. Gate must be in front of the camera  (tvec Z > 0)
    #   2. Panel must face the camera: rotate the panel normal [0,0,1] into
    #      camera frame via R = Rodrigues(rvec).  If the panel faces us, the
    #      rotated normal points back toward the camera → normal[2] < 0.
    # If both solutions pass (rare), fall back to the one closest to depth_ref.
    valid = []
    for i in range(retval):
        if float(tvecs[i][2][0]) <= 0:
            continue
        R_sol  = cv2.Rodrigues(rvecs[i])[0]
        normal = R_sol @ np.array([0.0, 0.0, 1.0])
        if normal[2] < 0:          # panel normal points toward camera ✓
            valid.append(i)

    if not valid:
        return None
    best = min(valid, key=lambda i: abs(float(tvecs[i][2][0]) - depth_ref))
    tvec = tvecs[best].flatten()   # gate centre in OpenCV camera frame

    # OpenCV camera frame: X=right, Y=down, Z=forward (into scene)
    # Body frame used here: X=forward, Y=left, Z=up
    pos_cam  = tvec
    pos_body = np.array([pos_cam[2], -pos_cam[0], -pos_cam[1]])

    # Rotate from body frame to world frame using drone attitude
    roll, pitch, yaw = sensor_data['roll'], sensor_data['pitch'], sensor_data['yaw']
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    R_bw = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])

    drone_pos = np.array([sensor_data['x_global'],
                          sensor_data['y_global'],
                          sensor_data['z_global']])
    return drone_pos + R_bw @ pos_body


# ---------------------------------------------------------------------------
# Outlier-robust position estimate
# ---------------------------------------------------------------------------
def _estimate_position(observations):
    """
    Given a list of (world_pos, area) observations for one gate, return the
    area-weighted mean position after MAD-based outlier removal.

    observations : list of (np.ndarray shape(3,), float)
    Returns      : np.ndarray shape(3,)
    """
    if len(observations) == 1:
        return observations[0][0].copy()

    positions = np.array([p for p, _ in observations])   # (N, 3)
    areas     = np.array([a for _, a in observations])   # (N,)

    # Median position (component-wise)
    median = np.median(positions, axis=0)

    # Distance of each observation from the median
    dists = np.linalg.norm(positions - median, axis=1)

    # Median Absolute Deviation
    mad = float(np.median(dists))

    # Inlier threshold: at least 0.30 m, or 3× MAD
    threshold = max(3.0 * mad, 0.30)

    inlier_mask = dists <= threshold
    if not np.any(inlier_mask):        # all rejected → keep all
        inlier_mask = np.ones(len(observations), dtype=bool)

    in_pos  = positions[inlier_mask]
    in_area = areas[inlier_mask]

    w_sum = in_area.sum()
    return np.dot(in_area, in_pos) / w_sum


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class MyAssignment:

    # State machine constants
    _S_APPROACH = 'approach'
    _S_SEARCH   = 'search'
    _S_DONE     = 'done'

    def __init__(self):
        # Detection – one entry per gate:
        #   {'observations': [(world_pos, area), ...], 'pos': np.array, 'best_area': float}
        self._gates = []

        # Navigation state machine
        self._nav_state       = self._S_SEARCH
        self._gates_passed    = set()
        self._traj            = None   # (N, 3) discretised trajectory
        self._traj_idx        = 0
        self._target_gate     = None   # index into self._gates
        self._centre_traj_idx = None   # traj index where gate is considered passed
        self._search_yaw      = 0.0

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _match_gate(self, world_pos):
        for i, g in enumerate(self._gates):
            if np.linalg.norm(world_pos[:2] - g['pos'][:2]) < _SAVE_DEDUP_RADIUS:
                return i
        return -1

    def _update_detections(self, bgr, sensor_data):
        """Run detection on bgr and update the gate registry."""
        detections = detect_gate_corners(bgr)
        for corners in detections:
            if (corners[:, 0].min() < _FULL_PANEL_MARGIN or
                    corners[:, 0].max() > IMG_W - _FULL_PANEL_MARGIN or
                    corners[:, 1].min() < _FULL_PANEL_MARGIN or
                    corners[:, 1].max() > IMG_H - _FULL_PANEL_MARGIN):
                continue

            world_pos = pnp_gate_pose(corners, sensor_data)
            if world_pos is None:
                continue

            area = float(cv2.contourArea(corners))
            idx  = self._match_gate(world_pos)

            if idx < 0:
                if len(self._gates) >= NUM_GATES:
                    continue
                self._gates.append({'observations': [(world_pos.copy(), area)],
                                    'pos': world_pos.copy(),
                                    'best_area': area})
                gate_num = len(self._gates)
                self._save_image(bgr, corners, world_pos, gate_num)
                print(f"[Gate {gate_num}] first detection  "
                      f"pos=({world_pos[0]:.2f},{world_pos[1]:.2f},{world_pos[2]:.2f})  "
                      f"area={area:.0f}px²")
                if gate_num == NUM_GATES:
                    _save_results(self._gates)
            else:
                g = self._gates[idx]
                g['observations'].append((world_pos.copy(), area))
                g['pos'] = _estimate_position(g['observations'])
                gate_num = idx + 1
                n_obs    = len(g['observations'])
                print(f"[Gate {gate_num}] updated ({n_obs} obs)  "
                      f"pos=({g['pos'][0]:.2f},{g['pos'][1]:.2f},{g['pos'][2]:.2f})  "
                      f"area={area:.0f}px²")
                if area > g['best_area']:
                    g['best_area'] = area
                    self._save_image(bgr, corners, g['pos'], gate_num)
                if len(self._gates) == NUM_GATES:
                    _save_results(self._gates)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _nearest_unvisited(self, drone_xy):
        """Return the index of the nearest known gate not yet passed, or None."""
        best_idx  = None
        best_dist = float('inf')
        for i, g in enumerate(self._gates):
            if i in self._gates_passed:
                continue
            d = float(np.linalg.norm(drone_xy - g['pos'][:2]))
            if d < best_dist:
                best_dist = d
                best_idx  = i
        return best_idx

    def _poly_matrix(self, t):
        """6×6 constraint matrix for a 5th-order polynomial segment of duration t."""
        T = float(t)
        return np.array([
            [1,  0,    0,      0,       0,        0      ],  # p(0)
            [0,  1,    0,      0,       0,        0      ],  # p'(0)
            [0,  0,    2,      0,       0,        0      ],  # p''(0)
            [1,  T,    T**2,   T**3,    T**4,     T**5   ],  # p(T)
            [0,  1,    2*T,    3*T**2,  4*T**3,   5*T**4 ],  # p'(T)
            [0,  0,    2,      6*T,    12*T**2,  20*T**3 ],  # p''(T)
        ])

    def _min_jerk_trajectory(self, waypoints):
        """
        Build a minimum-jerk trajectory through *waypoints* (list of [x,y,z]).
        Each segment uses a 5th-order polynomial with zero velocity and
        acceleration at both endpoints.

        Returns an (N, 3) ndarray of discretised positions.
        """
        M   = self._poly_matrix(T_SEG)
        pts = []
        n   = len(waypoints)

        for i in range(n - 1):
            p0 = np.array(waypoints[i][:3],   dtype=float)
            p1 = np.array(waypoints[i + 1][:3], dtype=float)
            # include endpoint only for the last segment to avoid duplicates
            last_seg = (i == n - 2)
            ts = np.linspace(0.0, T_SEG, DISC_STEPS, endpoint=last_seg)

            seg = np.zeros((len(ts), 3))
            for dim in range(3):
                b = np.array([p0[dim], 0.0, 0.0, p1[dim], 0.0, 0.0])
                c = np.linalg.solve(M, b)
                for j, t in enumerate(ts):
                    seg[j, dim] = (c[0] + c[1]*t + c[2]*t**2
                                   + c[3]*t**3 + c[4]*t**4 + c[5]*t**5)
            pts.append(seg)

        return np.vstack(pts)

    def _plan_to_gate(self, gate_idx, drone_xyz):
        """Build and store a min-jerk trajectory toward gate *gate_idx*."""
        gate_pos = self._gates[gate_idx]['pos']
        z = float(gate_pos[2]) if not np.isnan(float(gate_pos[2])) else CRUISE_Z

        to_gate = gate_pos[:2] - drone_xyz[:2]
        dist    = float(np.linalg.norm(to_gate))
        d_hat   = to_gate / dist if dist > 0.1 else np.array([1.0, 0.0])

        ap   = gate_pos[:2] - d_hat * APPROACH_DIST   # approach point
        past = gate_pos[:2] + d_hat * PAST_DIST        # pull-through point

        waypoints = [
            [drone_xyz[0], drone_xyz[1], drone_xyz[2]],
            [ap[0],        ap[1],        z            ],
            [gate_pos[0],  gate_pos[1],  z            ],
            [past[0],      past[1],      z            ],
        ]

        traj = self._min_jerk_trajectory(waypoints)
        self._traj            = traj
        self._traj_idx        = 0
        # Gate is considered passed once we reach the end of the trajectory
        # (which is PAST_DIST beyond the gate centre).
        self._centre_traj_idx = len(traj) - 1

    # ------------------------------------------------------------------
    # Navigation state machine
    # ------------------------------------------------------------------

    def _navigate(self, sensor_data):
        drone_xyz = np.array([sensor_data['x_global'],
                              sensor_data['y_global'],
                              sensor_data['z_global']])
        drone_xy  = drone_xyz[:2]
        yaw       = float(sensor_data['yaw'])

        # ── DONE ──────────────────────────────────────────────────────────
        if self._nav_state == self._S_DONE:
            return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]

        # ── APPROACH: follow trajectory, check if gate is passed ──────────
        if self._nav_state == self._S_APPROACH and self._traj is not None:
            pt   = self._traj[self._traj_idx]
            dist = float(np.linalg.norm(drone_xyz - pt))
            if dist < TRAJ_STEP_THRESH and self._traj_idx < len(self._traj) - 1:
                self._traj_idx += 1

            if self._traj_idx >= self._centre_traj_idx:
                # Gate passed
                self._gates_passed.add(self._target_gate)
                print(f"[Nav] gate {self._target_gate + 1} passed "
                      f"({len(self._gates_passed)}/{NUM_GATES})")
                if len(self._gates_passed) >= NUM_GATES:
                    self._nav_state = self._S_DONE
                    print("[Nav] all gates passed → DONE")
                    return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]
                self._nav_state = self._S_SEARCH
                self._search_yaw = yaw
                print("[Nav] → SEARCH")
                # fall through to SEARCH below
            else:
                pt = self._traj[self._traj_idx]
                if self._traj_idx < len(self._traj) - 1:
                    nxt = self._traj[self._traj_idx + 1]
                    dx, dy = nxt[0] - pt[0], nxt[1] - pt[1]
                    if abs(dx) + abs(dy) > 0.01:
                        yaw = float(np.arctan2(dy, dx))
                return [float(pt[0]), float(pt[1]), float(pt[2]), yaw]

        # ── SEARCH: rotate until a gate is visible, then approach ─────────
        if self._nav_state == self._S_SEARCH:
            next_gate = self._nearest_unvisited(drone_xy)
            if next_gate is not None:
                self._target_gate = next_gate
                self._plan_to_gate(next_gate, drone_xyz)
                self._nav_state = self._S_APPROACH
                print(f"[Nav] target gate {next_gate + 1} → APPROACH")
                pt = self._traj[0]
                return [float(pt[0]), float(pt[1]), float(pt[2]), yaw]
            # No gate known yet – rotate in place
            self._search_yaw += SEARCH_YAW_RATE
            return [drone_xyz[0], drone_xyz[1], CRUISE_Z, self._search_yaw]

        # Fallback
        return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute_command(self, sensor_data, camera_data, _dt):
        if sensor_data['z_global'] < 0.49:
            return [sensor_data['x_global'], sensor_data['y_global'],
                    1.0, sensor_data['yaw']]

        bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        self._update_detections(bgr, sensor_data)
        return self._navigate(sensor_data)

    @staticmethod
    def _save_image(bgr, corners, world_pos, gate_num):
        out = bgr.copy()

        # Draw detected corners
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in corners:
            cv2.circle(out, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

        # Draw sorted corners to show IPPE ordering (TL/TR/BR/BL)
        sorted_c = _sort_corners_tl_tr_br_bl(corners)
        labels   = ['TL', 'TR', 'BR', 'BL']
        for pt, lbl in zip(sorted_c, labels):
            cv2.putText(out, lbl, (int(pt[0]) + 4, int(pt[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # Overlay estimated world position
        label = (f"Gate {gate_num}  "
                 f"({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}) m")
        cv2.putText(out, label, (4, out.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        path = os.path.join(_IMG_DIR, f'gate_{gate_num}.png')
        cv2.imwrite(path, out)
        print(f"[Save] gate_{gate_num}.png  pos=({world_pos[0]:.2f},"
              f"{world_pos[1]:.2f},{world_pos[2]:.2f})")


def _save_results(gates):
    """Write estimated vs ground-truth gate positions to _RESULTS_FILE."""
    import json

    truth = []
    if os.path.exists(_TRUTH_FILE):
        with open(_TRUTH_FILE) as f:
            truth = json.load(f)

    lines = ['Gate  |   Estimated (x, y, z)          |   Ground truth (x, y, z)       |  Error (m)  |  Obs',
             '-' * 100]
    for i, g in enumerate(gates):
        est     = g['pos']
        est_str = f'({est[0]:6.3f}, {est[1]:6.3f}, {est[2]:6.3f})'
        if i < len(truth):
            gt      = truth[i]
            err     = float(np.linalg.norm(est - np.array([gt['x'], gt['y'], gt['z']])))
            gt_str  = f'({gt["x"]:6.3f}, {gt["y"]:6.3f}, {gt["z"]:6.3f})'
            err_str = f'{err:.3f}'
        else:
            gt_str  = 'N/A'
            err_str = 'N/A'
        n_obs = len(g['observations'])
        lines.append(f'  {i+1}   |  {est_str}  |  {gt_str}  |  {err_str:>9}  |  {n_obs} obs')

    with open(_RESULTS_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'[Results] {_RESULTS_FILE}')
    for line in lines:
        print(line)


_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
