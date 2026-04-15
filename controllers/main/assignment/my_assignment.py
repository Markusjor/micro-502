import os

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

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
ARENA_CENTER     = np.array([4.0, 4.0])  # gates arranged tangentially around this
CRUISE_Z         = 1.0    # m – default flight altitude
APPROACH_DIST    = 0.4    # m – approach point in front of gate on its normal
PAST_DIST        = 1.2    # m – how far past gate centre before gate is considered passed
T_APPROACH       = 2.5    # s – polynomial duration for flying to approach point
T_THROUGH        = 1.5    # s – polynomial duration for flying through gate
T_REPLAN_BUF     = 0.4    # s – time-ahead buffer: replan from polynomial state this far ahead
REPLAN_INTERVAL  = 0.25   # s – replan approach polynomial at least this often
APPROACH_TOL     = 0.06   # m – arrival radius at approach point → switch to ALIGN
ALIGN_TOL        = 0.10   # m – position must stay within this during ALIGN (else counter resets)
YAW_TOL          = 0.12   # rad – yaw must be within this before ALIGN counter advances
ALIGN_STEPS      = 80     # consecutive on-target control steps required before flying through
YAW_RATE_MAX     = 0.06   # rad/step – max yaw rate during alignment
SEARCH_YAW_RATE  = 0.04   # rad/step
SEARCH_Z_AMP     = 0.20   # m – vertical oscillation amplitude during search
SEARCH_Z_RATE    = 0.06   # rad/step for vertical oscillation
SIDE_STEP_DIST   = 2.0    # m – lateral offset (rightward) after gate pass before rotating
SIDE_STEP_TOL    = 0.15   # m – arrival threshold for the side-step waypoint

# Lap spline (laps 2 and 3)
# One knot per gate (gate centre).  Periodic CubicSpline through the 5 gate
# centres produces the smooth oval racing line naturally without any
# artificial before/after tangent constraints.
LAP_SEG_T        = 2.0    # s per gate-to-gate spline segment (spline parameterisation only)
LAP_SPEED        = 3.5    # m/s target speed during laps 2/3 (raise for more risk/speed)
LAP_LOOKAHEAD_T  = 0.8    # s of spline ahead to use as tracking target
LAP_POLY_MIN_T   = 0.25   # minimum polynomial duration for lap replans
LAP_SEARCH_AHEAD = LAP_SEG_T * 1.5 # s of spline to search when projecting drone position

# CCW sector ordering constraint.
# Each gate's angular position (atan2 from ARENA_CENTER) is used to determine
# whether it could be the immediately next gate in the circuit.
#
# Primary check: the candidate's CCW advance from the last passed gate must be
# ≤ (drone's own CCW advance) + one sector (360/N).  This uses the drone's
# current global position to ask "is there room for an undetected gate between
# me and this candidate?"  If the candidate is more than one sector ahead of
# where the drone currently is, there likely IS an undetected gate in between.
#
# When no gate passes the primary check the drone flies toward the estimated
# position of the next sector so it can physically search there.  After
# SECTOR_FALLBACK_STEPS of searching without success the constraint is relaxed
# to SECTOR_WIDE_DEG to recover from unusual spacings.
SECTOR_MIN_DEG      = 15     # deg – minimum CCW advance (prevents going backward)
SECTOR_WIDE_DEG     = 200    # deg – fallback maximum after extended searching
SECTOR_FALLBACK_STEPS = 200  # search steps before switching to wide window



# ---------------------------------------------------------------------------
# Polynomial planner helpers
# ---------------------------------------------------------------------------
def _poly5_fit(p0, v0, a0, p1, v1, a1, T):
    """
    Fit a 5th-order polynomial for each spatial dimension.
    Boundary conditions: position, velocity, acceleration at t=0 and t=T.
    Returns (3, 6) coefficient array c such that x(t) = c @ [1,t,t²,t³,t⁴,t⁵].
    """
    T = max(float(T), 0.05)
    M = np.array([
        [1,  0,    0,      0,       0,        0      ],
        [0,  1,    0,      0,       0,        0      ],
        [0,  0,    2,      0,       0,        0      ],
        [1,  T,    T**2,   T**3,    T**4,     T**5   ],
        [0,  1,    2*T,    3*T**2,  4*T**3,   5*T**4 ],
        [0,  0,    2,      6*T,    12*T**2,  20*T**3 ],
    ])
    p0, v0, a0 = np.asarray(p0, float), np.asarray(v0, float), np.asarray(a0, float)
    p1, v1, a1 = np.asarray(p1, float), np.asarray(v1, float), np.asarray(a1, float)
    coeff = np.zeros((3, 6))
    for d in range(3):
        coeff[d] = np.linalg.solve(M, [p0[d], v0[d], a0[d], p1[d], v1[d], a1[d]])
    return coeff


def _poly5_eval(coeff, tau, T):
    """
    Evaluate position, velocity, and acceleration of a 5th-order polynomial.
    tau is clamped to [0, T].  Returns (pos, vel, acc) each shape (3,).
    """
    tau = float(np.clip(tau, 0.0, float(T)))
    t2, t3, t4, t5 = tau**2, tau**3, tau**4, tau**5
    pos = coeff @ np.array([1,   tau,    t2,      t3,       t4,       t5    ])
    vel = coeff @ np.array([0,   1,      2*tau,   3*tau**2, 4*tau**3, 5*tau**4])
    acc = coeff @ np.array([0,   0,      2,       6*tau,   12*tau**2, 20*tau**3])
    return pos, vel, acc


def _angle_wrap(a):
    """Wrap angle to [-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


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
    _S_LAP      = 'lap'
    _S_DONE     = 'done'

    def __init__(self):
        # Detection – one entry per gate:
        #   {'observations': [(world_pos, area), ...], 'pos': np.array, 'best_area': float}
        self._gates = []

        # Navigation state machine
        self._nav_state     = self._S_SEARCH
        self._gates_passed           = set()
        self._target_gate            = None
        self._last_gate_angle        = None  # angle from ARENA_CENTER of last passed gate (rad)
        self._search_yaw             = 0.0
        self._search_step            = 0
        self._search_steps_no_target = 0     # increments while SEARCH finds no valid gate
        self._search_side_target     = None  # brief rightward move after gate pass

        # Approach geometry (set by _plan_to_gate, fixed for current gate)
        self._gate_ap       = None   # approach point [x,y,z] – 0.4m in front of gate
        self._gate_past     = None   # past point [x,y,z] – PAST_DIST beyond gate
        self._gate_d_hat    = None   # unit vector: approach direction (gate's outward normal)
        self._gate_yaw      = 0.0   # yaw angle that faces the gate

        # Sub-phase within APPROACH: 0=fly-to-AP, 1=align-yaw, 2=through-gate
        self._ap_phase      = 0
        self._align_count   = 0

        # Smooth polynomial plan state
        self._plan_coeff    = None   # (3,6) 5th-order polynomial coefficients, or None
        self._plan_t0       = 0.0   # sim time when current polynomial was fitted
        self._plan_T        = 1.0   # duration of current polynomial
        self._last_replan   = -999.0
        self._sim_time      = 0.0   # accumulated dt (seconds)

        # Lap spline (built after all gates passed in lap 1)
        self._gates_order    = []    # gate indices in the order they were passed
        self._lap_spline     = None  # CubicSpline (position vs knot time)
        self._lap_spline_vel = None  # first derivative of _lap_spline
        self._lap_total_T    = 0.0   # period of one lap on the spline (s)
        self._lap_t_progress = 0.0   # current closest-point parameter along spline

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
        """
        Return the index of the next gate to target in CCW order, or None.

        After the first gate is passed the method uses the drone's current global
        position to enforce a single-sector lookahead:

          gate_advance  = CCW angle from last-passed-gate to candidate
          drone_advance = CCW angle from last-passed-gate to drone's current position

        A candidate is accepted only when:
          gate_advance  ≤  drone_advance + sector_size          (primary)

        This rejects a gate that is more than one sector ahead of where the drone
        currently is, preventing a gate that is two CCW slots away from being
        targeted before the closer (undetected) intermediate gate is found.

        After SECTOR_FALLBACK_STEPS search steps without any valid target the
        window relaxes to SECTOR_WIDE_DEG so the drone recovers from layouts
        where the true next gate genuinely has an above-average angular gap.

        Among all passing candidates the one with the smallest CCW advance is
        returned – strict ordering regardless of physical distance.
        """
        unvisited = [i for i in range(len(self._gates)) if i not in self._gates_passed]
        if not unvisited:
            return None

        sector_rad  = 2.0 * np.pi / NUM_GATES   # one sector ≈ 72° for N=5
        lo          = np.radians(SECTOR_MIN_DEG)

        # Reference angle for the CCW ordering constraint.
        # After the first gate: use that gate's angular position.
        # Before any gate is passed: use the drone's own angular position so the
        # sector window is anchored at the drone rather than skipped entirely.
        # This prevents targeting a gate that is angularly "behind" the drone
        # (e.g. gate 5 when gate 1 is the natural next CCW gate).
        drone_angle = float(np.arctan2(drone_xy[1] - ARENA_CENTER[1],
                                       drone_xy[0] - ARENA_CENTER[0]))
        ref_angle   = self._last_gate_angle if self._last_gate_angle is not None else drone_angle

        # Drone's CCW advance from the reference angle (0 when ref = drone position)
        drone_advance = (drone_angle - ref_angle) % (2.0 * np.pi)

        # Primary: gate must be within one sector ahead of the drone's position
        hi_primary = drone_advance + sector_rad
        # Fallback: after extended search use the wide fixed window
        hi_fallback = np.radians(SECTOR_WIDE_DEG)
        use_fallback = self._search_steps_no_target >= SECTOR_FALLBACK_STEPS
        hi = hi_fallback if use_fallback else hi_primary

        candidates = []
        for i in unvisited:
            pos     = self._gates[i]['pos']
            angle   = float(np.arctan2(pos[1] - ARENA_CENTER[1],
                                       pos[0] - ARENA_CENTER[0]))
            advance = (angle - ref_angle) % (2.0 * np.pi)
            if lo <= advance <= hi:
                candidates.append((i, advance))

        if not candidates:
            return None

        # Strict CCW ordering: pick smallest advance (not nearest by distance)
        return min(candidates, key=lambda x: x[1])[0]

    def _plan_to_gate(self, gate_idx, drone_xyz):
        """
        Select approach geometry for gate *gate_idx* and fit the initial polynomial.

        The approach axis is tangential to the CCW circuit around ARENA_CENTER.
        The approach point (AP) is placed APPROACH_DIST metres in front of the gate
        on the side closest to the drone so we never fly around the back.
        """
        gate_pos = self._gates[gate_idx]['pos']
        z = float(gate_pos[2]) if not np.isnan(float(gate_pos[2])) else CRUISE_Z

        radial = gate_pos[:2] - ARENA_CENTER
        r_len  = float(np.linalg.norm(radial))
        if r_len > 0.2:
            r_hat   = radial / r_len
            tangent = np.array([-r_hat[1], r_hat[0]])   # 90° CCW = tangential (gate normal)
            ap_a = gate_pos[:2] - tangent * APPROACH_DIST
            ap_b = gate_pos[:2] + tangent * APPROACH_DIST
            drone_xy = drone_xyz[:2]
            if np.linalg.norm(drone_xy - ap_a) <= np.linalg.norm(drone_xy - ap_b):
                d_hat, ap_xy = tangent, ap_a
            else:
                d_hat, ap_xy = -tangent, ap_b
        else:
            to_gate = gate_pos[:2] - drone_xyz[:2]
            dist    = float(np.linalg.norm(to_gate))
            d_hat   = to_gate / dist if dist > 0.1 else np.array([1.0, 0.0])
            ap_xy   = gate_pos[:2] - d_hat * APPROACH_DIST

        past_xy = gate_pos[:2] + d_hat * PAST_DIST

        self._gate_ap    = np.array([ap_xy[0],   ap_xy[1],   z])
        self._gate_past  = np.array([past_xy[0], past_xy[1], z])
        self._gate_d_hat = d_hat.copy()
        self._gate_yaw   = float(np.arctan2(d_hat[1], d_hat[0]))
        self._ap_phase   = 0
        self._align_count = 0

        # Fit initial polynomial from current drone position (at rest) to AP.
        self._plan_coeff  = _poly5_fit(
            drone_xyz, np.zeros(3), np.zeros(3),
            self._gate_ap, np.zeros(3), np.zeros(3),
            T_APPROACH,
        )
        self._plan_t0     = self._sim_time
        self._plan_T      = T_APPROACH
        self._last_replan = self._sim_time

    def _replan_to(self, target_xyz, duration=T_APPROACH, target_vel=None):
        """
        Smooth replan toward target_xyz.

        Samples the current polynomial T_REPLAN_BUF seconds ahead to obtain the
        predicted (position, velocity, acceleration), then fits a new polynomial
        from that predicted state to target_xyz.  This guarantees C2 continuity
        across replans — no teleporting or velocity discontinuities.

        target_vel : desired velocity at target_xyz (None → zero, use non-zero for
                     waypoints that should be passed through without stopping).
        """
        t_eval = min(self._sim_time + T_REPLAN_BUF,
                     self._plan_t0 + self._plan_T)
        tau    = t_eval - self._plan_t0
        p0, v0, a0 = _poly5_eval(self._plan_coeff, tau, self._plan_T)

        v1 = np.zeros(3) if target_vel is None else np.asarray(target_vel, float)
        self._plan_coeff  = _poly5_fit(p0, v0, a0, target_xyz, v1, np.zeros(3), duration)
        self._plan_t0     = t_eval
        self._plan_T      = duration
        self._last_replan = self._sim_time

    def _build_lap_spline(self, start_xyz):
        """
        Fit a periodic cubic spline through the gate centres in pass order.

        One knot per gate (the gate centre position).  A periodic CubicSpline
        through N gate centres naturally produces the smooth oval racing line
        without imposing artificial before/after tangent constraints that would
        introduce oscillations or detours.

        The spline and its derivative are stored; the active polynomial is seeded
        from the drone's current position to knot-0 so _replan_to has a valid
        state to sample from on the very first control step.
        """
        waypoints = []
        for gate_idx in self._gates_order:
            gate_pos = self._gates[gate_idx]['pos']
            z = float(gate_pos[2]) if not np.isnan(float(gate_pos[2])) else CRUISE_Z
            waypoints.append(np.array([float(gate_pos[0]), float(gate_pos[1]), z]))

        N = len(waypoints)
        if N < 2:
            return

        # Uniform time knots; close the loop by repeating the first point.
        t_knots = np.arange(N + 1) * LAP_SEG_T
        pts     = np.vstack(waypoints + [waypoints[0]])   # (N+1, 3)

        # Periodic BC: equal first and second derivatives at the seam.
        self._lap_spline     = CubicSpline(t_knots, pts, bc_type='periodic')
        self._lap_spline_vel = self._lap_spline.derivative()
        self._lap_total_T    = float(N * LAP_SEG_T)
        self._lap_t_progress = 0.0

        # Seed the active polynomial from the drone toward the t=0 spline point.
        sv0    = self._lap_spline_vel(0.0)
        sp0    = self._lap_spline(0.0)
        dist   = float(np.linalg.norm(start_xyz - sp0))
        init_T = max(1.0, dist / 1.5)
        self._plan_coeff  = _poly5_fit(
            start_xyz, np.zeros(3), np.zeros(3),
            sp0, sv0, np.zeros(3),
            init_T,
        )
        self._plan_t0     = self._sim_time
        self._plan_T      = init_T
        self._last_replan = self._sim_time

        print(f"[Nav] cubic spline: {N} gate-centre knots, total_T={self._lap_total_T:.1f}s")
        self._plot_lap_spline(waypoints, start_xyz)

    def _plot_lap_spline(self, waypoints, start_xyz):
        """
        Open an interactive 3-D figure of the planned lap path in a separate process.

        Matplotlib's GUI event loop must run on a main thread.  Spawning a fresh
        process gives the plot its own main thread so the window is fully
        interactive (rotate, zoom, pan) without blocking the planner.
        """
        import multiprocessing

        total_T = len(waypoints) * LAP_SEG_T
        t_dense = np.linspace(0, total_T, 400)
        curve   = self._lap_spline(t_dense)          # (400, 3) ndarray

        gate_centres = [
            (int(gi + 1), [float(v) for v in self._gates[gi]['pos']])
            for gi in self._gates_order
        ]

        p = multiprocessing.Process(
            target=_spline_plot_process,
            args=(curve.tolist(),
                  [wp.tolist() for wp in waypoints],
                  gate_centres,
                  [float(v) for v in start_xyz]),
            daemon=True,
        )
        p.start()

    # ------------------------------------------------------------------
    # Navigation state machine
    # ------------------------------------------------------------------

    def _navigate(self, sensor_data):
        drone_xyz = np.array([sensor_data['x_global'],
                              sensor_data['y_global'],
                              sensor_data['z_global']])
        drone_xy = drone_xyz[:2]
        yaw      = float(sensor_data['yaw'])

        # ── DONE ──────────────────────────────────────────────────────────
        if self._nav_state == self._S_DONE:
            return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]

        # ── LAP: continuously-updated closest-point tracking on cubic spline ──
        if self._nav_state == self._S_LAP:
            if self._lap_spline is None:
                return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]

            # ── 1. Project drone onto spline (closest-point advance) ──────────
            # Sample the spline from the current progress up to LAP_SEARCH_AHEAD
            # seconds ahead, find the closest sample, and advance _lap_t_progress.
            # Searching only forward prevents the progress from drifting backward.
            t_lo      = self._lap_t_progress
            t_hi      = t_lo + LAP_SEARCH_AHEAD
            t_samples = np.linspace(t_lo, t_hi, 60) % self._lap_total_T
            pts       = self._lap_spline(t_samples)               # (60, 3)
            best      = int(np.argmin(np.linalg.norm(pts - drone_xyz, axis=1)))
            self._lap_t_progress = float(
                np.linspace(t_lo, t_hi, 60)[best] % self._lap_total_T
            )

            # ── 2. Lookahead target on the spline ─────────────────────────────
            t_ref   = (self._lap_t_progress + LAP_LOOKAHEAD_T) % self._lap_total_T
            ref_pos = self._lap_spline(t_ref)
            ref_vel = self._lap_spline_vel(t_ref)

            # ── 3. Replan toward the (continuously updated) target ────────────
            # Same mechanism as phase-0 approach on lap 1: _replan_to fires every
            # REPLAN_INTERVAL, samples the current polynomial T_REPLAN_BUF ahead
            # for C2 continuity, and fits a new polynomial to ref_pos / ref_vel.
            if self._sim_time - self._last_replan > REPLAN_INTERVAL:
                dist_to_ref = float(np.linalg.norm(drone_xyz - ref_pos))
                seg_T = max(LAP_POLY_MIN_T, dist_to_ref / LAP_SPEED)
                self._replan_to(ref_pos, duration=seg_T, target_vel=ref_vel)

            tau = self._sim_time - self._plan_t0
            sp, vel, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)

            # Yaw from polynomial velocity; fall back to spline tangent when slow.
            vx, vy = float(vel[0]), float(vel[1])
            if abs(vx) > 0.05 or abs(vy) > 0.05:
                yaw_cmd = float(np.arctan2(vy, vx))
            else:
                svx, svy = float(ref_vel[0]), float(ref_vel[1])
                yaw_cmd = float(np.arctan2(svy, svx)) if (abs(svx) > 0.05 or abs(svy) > 0.05) else yaw
            return [float(sp[0]), float(sp[1]), float(sp[2]), yaw_cmd]

        # ── APPROACH ──────────────────────────────────────────────────────
        if self._nav_state == self._S_APPROACH and self._gate_d_hat is not None:
            ap_xy = self._gate_ap[:2]
            ap_z  = float(self._gate_ap[2])

            # ── Phase 0: smooth polynomial to approach point ───────────────
            if self._ap_phase == 0:
                # Keep AP position in sync with the latest gate position estimate
                gate_pos  = self._gates[self._target_gate]['pos']
                z_new     = float(gate_pos[2]) if not np.isnan(float(gate_pos[2])) else CRUISE_Z
                ap_xy_new = gate_pos[:2] - self._gate_d_hat * APPROACH_DIST
                ap_xyz_new = np.array([ap_xy_new[0], ap_xy_new[1], z_new])

                # Replan if AP shifted or enough time has elapsed since last replan.
                # The replan samples the current polynomial T_REPLAN_BUF s ahead and
                # fits a new polynomial from that predicted (p, v, a) – guaranteeing
                # C2 continuity so the drone never feels a jerk.
                ap_shift = float(np.linalg.norm(ap_xyz_new[:2] - self._gate_ap[:2]))
                if (self._sim_time - self._last_replan > REPLAN_INTERVAL
                        or ap_shift > 0.05):
                    self._gate_ap   = ap_xyz_new
                    self._gate_past = np.array([
                        gate_pos[0] + self._gate_d_hat[0] * PAST_DIST,
                        gate_pos[1] + self._gate_d_hat[1] * PAST_DIST,
                        z_new,
                    ])
                    self._replan_to(self._gate_ap)
                    ap_xy = self._gate_ap[:2]
                    ap_z  = float(self._gate_ap[2])

                # Evaluate setpoint from the continuous polynomial
                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)

                # Transition when drone is close enough to AP
                if float(np.linalg.norm(drone_xy - ap_xy)) < APPROACH_TOL:
                    self._ap_phase  = 1
                    self._align_count = 0
                    print(f"[Nav] gate {self._target_gate + 1} AP reached → ALIGN")

                return [float(sp[0]), float(sp[1]), float(sp[2]), self._gate_yaw]

            # ── Phase 1: hold at AP, align yaw, wait until settled ─────────
            if self._ap_phase == 1:
                yaw_err = _angle_wrap(self._gate_yaw - yaw)
                yaw_cmd = yaw + float(np.clip(yaw_err, -YAW_RATE_MAX, YAW_RATE_MAX))

                # Only count steps where drone is on-target in BOTH position and
                # yaw; any drift resets the counter so we never fly through the
                # gate unless the drone is truly settled on the gate normal.
                pos_err = float(np.linalg.norm(drone_xy - ap_xy))
                if pos_err < ALIGN_TOL and abs(yaw_err) < YAW_TOL:
                    self._align_count += 1
                else:
                    self._align_count = 0

                if self._align_count >= ALIGN_STEPS:
                    self._ap_phase = 2
                    # Fit a new polynomial from AP straight through to past point.
                    # Starts at rest (v=0, a=0) so the drone launches smoothly.
                    self._plan_coeff = _poly5_fit(
                        self._gate_ap, np.zeros(3), np.zeros(3),
                        self._gate_past, np.zeros(3), np.zeros(3),
                        T_THROUGH,
                    )
                    self._plan_t0 = self._sim_time
                    self._plan_T  = T_THROUGH
                    print(f"[Nav] gate {self._target_gate + 1} aligned → THROUGH")

                return [float(ap_xy[0]), float(ap_xy[1]), ap_z, yaw_cmd]

            # ── Phase 2: fly through gate via polynomial ───────────────────
            if self._ap_phase == 2:
                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)

                # Gate passed when drone has moved far enough past the gate centre
                gate_pos = self._gates[self._target_gate]['pos']
                proj = float(np.dot(drone_xy - gate_pos[:2], self._gate_d_hat))

                if proj >= PAST_DIST * 0.7 or tau >= self._plan_T:
                    # Record the angle of this gate so CCW ordering can find the next
                    gp = self._gates[self._target_gate]['pos']
                    self._last_gate_angle = float(
                        np.arctan2(gp[1] - ARENA_CENTER[1], gp[0] - ARENA_CENTER[0])
                    )
                    self._gates_order.append(self._target_gate)
                    self._gates_passed.add(self._target_gate)
                    print(f"[Nav] gate {self._target_gate + 1} passed "
                          f"({len(self._gates_passed)}/{NUM_GATES})")
                    if len(self._gates_passed) >= NUM_GATES:
                        self._build_lap_spline(drone_xyz)
                        self._nav_state = self._S_LAP
                        print("[Nav] all gates passed → LAP")
                        return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]
                    self._nav_state  = self._S_SEARCH
                    self._search_yaw = yaw
                    self._search_step = 0
                    # Step right of travel direction so the next gate enters camera view
                    right = np.array([self._gate_d_hat[1], -self._gate_d_hat[0]])
                    side_xy = drone_xy + right * SIDE_STEP_DIST
                    self._search_side_target = np.array([
                        float(side_xy[0]), float(side_xy[1]),
                        float(self._gate_ap[2]),
                    ])
                    self._search_steps_no_target = 0
                    print("[Nav] → SEARCH (side-step)")
                    # fall through to SEARCH
                else:
                    return [float(sp[0]), float(sp[1]), float(sp[2]), self._gate_yaw]

        # ── SEARCH: look for next gate in CCW order ───────────────────────
        if self._nav_state == self._S_SEARCH:
            next_gate = self._nearest_unvisited(drone_xy)
            if next_gate is not None:
                self._target_gate            = next_gate
                self._search_steps_no_target = 0
                self._search_side_target     = None
                self._plan_to_gate(next_gate, drone_xyz)
                self._nav_state = self._S_APPROACH
                print(f"[Nav] target gate {next_gate + 1} → APPROACH")
                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)
                return [float(sp[0]), float(sp[1]), float(sp[2]), self._gate_yaw]

            # No qualifying gate found yet – rotate camera while moving/hovering.
            self._search_yaw  += SEARCH_YAW_RATE
            self._search_step += 1
            z_sp = CRUISE_Z + SEARCH_Z_AMP * np.sin(self._search_step * SEARCH_Z_RATE)

            # Phase A: brief rightward step after gate pass to open camera angle.
            # Do NOT fly CCW (sector-flying): that grows drone_advance and widens
            # the acceptance window, letting a gate two slots ahead slip through.
            if self._search_side_target is not None:
                dist = float(np.linalg.norm(drone_xy - self._search_side_target[:2]))
                if dist > SIDE_STEP_TOL:
                    return [self._search_side_target[0],
                            self._search_side_target[1],
                            z_sp, self._search_yaw]
                self._search_side_target = None   # arrived – switch to rotate-in-place

            # Phase B: rotate in place; increment no-target counter for fallback.
            self._search_steps_no_target += 1
            return [drone_xyz[0], drone_xyz[1], z_sp, self._search_yaw]

        # Fallback
        return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def compute_command(self, sensor_data, camera_data, dt):
        self._sim_time += float(dt)   # advance clock for polynomial evaluation

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


def _spline_plot_process(curve, waypoints, gate_centres, start_xyz):
    """
    Render an interactive 3-D spline figure.

    Runs in a dedicated child process so matplotlib's GUI event loop lives on
    that process's main thread.  All arguments are plain Python lists so they
    survive pickling across the process boundary.
    """
    import matplotlib.pyplot as plt

    curve = [tuple(p) for p in curve]
    xs, ys, zs = zip(*curve)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Spline path
    ax.plot(xs, ys, zs, color='steelblue', linewidth=2, label='spline path')

    # Knots (before / after each gate)
    wx = [p[0] for p in waypoints]
    wy = [p[1] for p in waypoints]
    wz = [p[2] for p in waypoints]
    ax.scatter(wx, wy, wz, color='limegreen', s=50, zorder=5, label='knots')

    # Gate centres + labels
    for label, pos in gate_centres:
        ax.scatter(pos[0], pos[1], pos[2],
                   color='red', marker='x', s=100, linewidths=2)
        ax.text(pos[0], pos[1], pos[2] + 0.10,
                f'G{label}', fontsize=9, ha='center', color='red')

    # Drone position at end of lap 1
    ax.scatter(start_xyz[0], start_xyz[1], start_xyz[2],
               color='orange', marker='*', s=150, zorder=6, label='lap 1 end')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Lap 2/3 planned path  –  drag to rotate')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.show()   # blocks until the window is closed; fine in a daemon process


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
