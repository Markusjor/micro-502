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
_SAVE_DEDUP_RADIUS  = 1.2   # m – match threshold for same-gate deduplication.
                             # PnP noise is ~0.1–0.3 m; 0.8 m gives margin without
                             # merging gates that are close together in tight seeds.
_FULL_PANEL_MARGIN  = 20   # px – reject if bounding box touches frame edge (partial panels give bad PnP)
NUM_GATES          = 5

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
ARENA_CENTER     = np.array([4.0, 4.0])  # gates arranged tangentially around this
ARENA_MARGIN     = 0.5                   # m – keep all waypoints this far from walls (0–8 m arena)
CRUISE_Z         = 1.0    # m – default flight altitude
APPROACH_DIST    = 0.4    # m – approach point in front of gate on its normal
PAST_DIST        = 1.2    # m – how far past gate centre before gate is considered passed
T_APPROACH       = 2.5    # s – polynomial duration for flying to approach point
T_THROUGH        = 1.5    # s – polynomial duration for flying through gate
T_REPLAN_BUF     = 0.2    # s – time-ahead buffer: replan from polynomial state this far ahead
REPLAN_INTERVAL  = 0.10   # s – replan approach polynomial at least this often
APPROACH_TOL     = 0.06   # m – arrival radius at approach point → switch to ALIGN
ALIGN_TOL        = 0.10   # m – position must stay within this during ALIGN (else counter resets)
YAW_TOL          = 0.12   # rad – yaw must be within this before ALIGN counter advances
ALIGN_STEPS      = 80     # consecutive on-target control steps required before flying through
MIN_GATE_OBS     = 200    # minimum observations required before flying through a gate
OBS_STALL_STEPS  = 25     # steps without new observations before starting yaw sweep
OBS_YAW_AMP      = 0.35   # rad – yaw sweep amplitude around gate_yaw
OBS_YAW_SPEED    = 0.04   # rad/step – sweep oscillation rate
YAW_RATE_MAX     = 0.06   # rad/step – max yaw rate during alignment
SEARCH_YAW_RATE  = 0.04   # rad/step
SEARCH_Z_AMP     = 0.20   # m – vertical oscillation amplitude during search
SEARCH_Z_RATE    = 0.06   # rad/step for vertical oscillation
SIDE_STEP_DIST   = 2.0    # m – total rightward movement after each gate pass
SIDE_STEP_SEGS   = 3      # number of equally-spaced waypoints in the side step
SIDE_STEP_TOL    = 0.20   # m – 2-D arrival radius per waypoint
SIDE_STEP_DUR    = 1.2    # s – polynomial duration per waypoint segment
# Lap spline (laps 2 and 3)
# One knot per gate (gate centre).  Periodic CubicSpline through the 5 gate
# centres produces the smooth oval racing line naturally without any
# artificial before/after tangent constraints.
LAP_SEG_T        = 2.0    # s per gate-to-gate spline segment (spline parameterisation only)
LAP_SPEED        = 6.0    # m/s fallback speed (used when velocity profile is unavailable)
LAP_LOOKAHEAD_T  = 0.8    # s of spline ahead – fallback when velocity profile is unavailable
LAP_POLY_MIN_T   = 0.15   # minimum polynomial duration for lap replans
LAP_SEARCH_AHEAD = LAP_SEG_T * 1.5 # s of spline to search when projecting drone position
# Curvature-adaptive velocity profile
LAP_A_MAX            = 4.0   # m/s² – max centripetal/longitudinal accel for velocity profile
LAP_V_MAX            = 13.0  # m/s – speed cap on straight sections
LAP_V_MIN            = 1.2   # m/s – speed floor even in the tightest turns
LAP_LOOKAHEAD_REAL_T = 0.65  # s  – real-flight-time lookahead horizon; scales with speed



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
    _S_APPROACH  = 'approach'
    _S_SIDE_STEP = 'side_step'
    _S_SEARCH    = 'search'
    _S_PRE_LAP   = 'pre_lap'   # fly to gate-1 entry before starting spline tracking
    _S_LAP       = 'lap'
    _S_DONE      = 'done'

    def __init__(self):
        # Detection – one entry per gate:
        #   {'observations': [(world_pos, area), ...], 'pos': np.array, 'best_area': float}
        self._gates = []

        # Navigation state machine
        self._nav_state     = self._S_SEARCH
        self._gates_passed  = set()
        self._target_gate   = None
        self._search_yaw    = 0.0
        self._search_step   = 0

        # Side-step state (rightward movement after each gate pass)
        self._side_step_wps  = []   # list of np.array([x,y,z]) waypoints
        self._side_step_idx  = 0    # index of the current target waypoint

        # Approach geometry (set by _plan_to_gate, fixed for current gate)
        self._gate_ap       = None   # approach point [x,y,z] – 0.4m in front of gate
        self._gate_past     = None   # past point [x,y,z] – PAST_DIST beyond gate
        self._gate_d_hat    = None   # unit vector: approach direction (gate's outward normal)
        self._gate_yaw      = 0.0   # yaw angle that faces the gate

        # Sub-phase within APPROACH: 0=fly-to-AP, 1=align-yaw, 2=through-gate
        self._ap_phase      = 0
        self._align_count   = 0

        # Observation-wait tracking (phase 1 → phase 2 gate)
        self._obs_last_n    = 0   # observation count the last time we checked
        self._obs_stall_step = 0  # steps elapsed without a new observation while waiting

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

        # Pre-lap entry point: approach waypoint before gate 1 at spline-tangent angle
        self._pre_lap_target = None  # np.array([x,y,z]) or None

        # Gate knot times on the lap spline (t values where gate centres sit).
        # Used to shrink lookahead near gates so corners are not cut.
        self._lap_gate_knots = []    # list of float knot times (one per real gate)

        # Curvature-adaptive velocity profile (built alongside _lap_spline)
        self._v_profile_spline = None

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

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _nearest_unvisited(self, drone_xy):
        """Return the nearest unvisited detected gate, or None."""
        unvisited = [i for i in range(len(self._gates)) if i not in self._gates_passed]
        if not unvisited:
            return None
        return min(unvisited,
                   key=lambda i: float(np.linalg.norm(drone_xy - self._gates[i]['pos'][:2])))

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
        self._obs_last_n  = 0
        self._obs_stall_step = 0

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

        # Insert a mid-arc waypoint on the closing segment (last gate → first gate)
        # to soften the entry angle on the spline seam and avoid steep gate misses.
        p_last  = waypoints[-1]
        p_first = waypoints[0]
        ang_last  = float(np.arctan2(p_last[1]  - ARENA_CENTER[1], p_last[0]  - ARENA_CENTER[0]))
        ang_first = float(np.arctan2(p_first[1] - ARENA_CENTER[1], p_first[0] - ARENA_CENTER[0]))
        ang_mid   = ang_last + _angle_wrap(ang_first - ang_last) * 0.5
        r_mid     = (float(np.linalg.norm(p_last[:2]  - ARENA_CENTER)) +
                     float(np.linalg.norm(p_first[:2] - ARENA_CENTER))) / 2.0
        z_mid     = (float(p_last[2]) + float(p_first[2])) / 2.0
        mid_xy    = ARENA_CENTER + r_mid * np.array([np.cos(ang_mid), np.sin(ang_mid)])
        waypoints.append(np.array([mid_xy[0], mid_xy[1], z_mid]))

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

        print(f"[Nav] cubic spline: {N} knots ({N-1} gates + 1 seam), total_T={self._lap_total_T:.1f}s")
        self._build_velocity_profile()
        self._plot_lap_spline(waypoints, start_xyz)

    def _build_velocity_profile(self):
        """
        Build a curvature-limited, trapezoidal velocity profile along the lap spline.

        Algorithm (forward-backward pass / "trapezoidal profile"):
          1. Sample curvature κ = |cs'×cs''| / |cs'|³ at N points.
          2. v_curvature = sqrt(a_max / κ)  – centripetal limit at each sample.
          3. Forward pass: limit each sample by the reachable speed from the
             previous sample given a_max over the arc-length step.
          4. Backward pass: same in reverse (deceleration).
          5. The array is tripled before the passes so the periodic boundary is
             handled correctly without a separate fixup.

        Stores self._v_profile_spline: CubicSpline mapping spline-parameter t
        → target flight speed (m/s).
        """
        N = 500
        t_samp = np.linspace(0.0, self._lap_total_T, N, endpoint=False)
        dt_samp = self._lap_total_T / N

        d1 = self._lap_spline(t_samp, 1)   # (N, 3)  m / spline-s
        d2 = self._lap_spline(t_samp, 2)   # (N, 3)

        # κ = |d1 × d2| / |d1|³
        cross  = np.cross(d1, d2)
        d1_mag = np.linalg.norm(d1, axis=1)
        kappa  = np.linalg.norm(cross, axis=1) / (d1_mag ** 3 + 1e-9)

        # Curvature-limited speed, clamped to [LAP_V_MIN, LAP_V_MAX]
        v_curv   = np.sqrt(np.maximum(LAP_A_MAX / (kappa + 1e-6), 0.0))
        v_profile = np.clip(v_curv, LAP_V_MIN, LAP_V_MAX)

        # Arc-length increment between consecutive samples (metres)
        ds = d1_mag * dt_samp   # ds[i] ≈ arc length from t_samp[i] to t_samp[i+1]

        # --- Trapezoidal passes on a tripled array (handles periodic BCs) ---
        ds3 = np.tile(ds, 3)
        v3  = np.tile(v_profile, 3)
        M   = 3 * N

        # Forward pass: v[i+1] ≤ sqrt(v[i]² + 2·a·ds[i])
        for i in range(M - 1):
            v_lim = np.sqrt(max(0.0, v3[i] ** 2 + 2.0 * LAP_A_MAX * ds3[i]))
            if v3[i + 1] > v_lim:
                v3[i + 1] = v_lim

        # Backward pass: v[i] ≤ sqrt(v[i+1]² + 2·a·ds[i])
        for i in range(M - 2, -1, -1):
            v_lim = np.sqrt(max(0.0, v3[i + 1] ** 2 + 2.0 * LAP_A_MAX * ds3[i]))
            if v3[i] > v_lim:
                v3[i] = v_lim

        # Middle period has both boundary conditions satisfied; re-apply floor
        v_profile = np.clip(v3[N:2 * N], LAP_V_MIN, LAP_V_MAX)

        self._v_profile_spline = CubicSpline(t_samp, v_profile, extrapolate=True)
        print(f"[VProfile] speed: min={v_profile.min():.2f}  "
              f"max={v_profile.max():.2f}  mean={v_profile.mean():.2f} m/s")

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

        # ── PRE_LAP: fly to gate-1 entry point before starting spline tracking ──
        if self._nav_state == self._S_PRE_LAP:
            if self._pre_lap_target is not None:
                dist = float(np.linalg.norm(drone_xyz - self._pre_lap_target))
                if dist < 0.4:
                    # Close enough — hand off to spline tracking
                    self._lap_t_progress = 0.0
                    self._nav_state = self._S_LAP
                    print("[Nav] PRE_LAP reached → LAP")
                else:
                    if self._sim_time - self._last_replan > REPLAN_INTERVAL:
                        self._replan_to(self._pre_lap_target,
                                        duration=max(1.0, dist / 1.5))
                    tau = self._sim_time - self._plan_t0
                    sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)
                    return [float(sp[0]), float(sp[1]), float(sp[2]), yaw]

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

            # ── 2. Adaptive speed and lookahead from curvature velocity profile ─
            t_curr = self._lap_t_progress % self._lap_total_T
            if self._v_profile_spline is not None:
                v_target = float(np.clip(
                    self._v_profile_spline(t_curr), LAP_V_MIN, LAP_V_MAX
                ))
            else:
                v_target = LAP_SPEED

            # Convert a fixed real-time lookahead (seconds) into spline-parameter
            # seconds via the local spline speed |cs'(t)| (metres / spline-s).
            # High speed on a straight → larger spline lookahead.
            # Low speed in a tight turn → smaller spline lookahead.
            spline_local_speed = float(np.linalg.norm(self._lap_spline_vel(t_curr)))
            if spline_local_speed > 0.05:
                lookahead_dist     = v_target * LAP_LOOKAHEAD_REAL_T   # metres
                lookahead_spline_t = float(np.clip(
                    lookahead_dist / spline_local_speed,
                    0.15, LAP_SEARCH_AHEAD * 0.8,
                ))
            else:
                lookahead_spline_t = LAP_LOOKAHEAD_T

            # ── 3. Lookahead target on the spline ─────────────────────────────
            t_ref   = (self._lap_t_progress + lookahead_spline_t) % self._lap_total_T
            ref_pos = self._lap_spline(t_ref)
            ref_vel = self._lap_spline_vel(t_ref)

            # ── 4. Replan toward the (continuously updated) target ────────────
            # Same mechanism as phase-0 approach on lap 1: _replan_to fires every
            # REPLAN_INTERVAL, samples the current polynomial T_REPLAN_BUF ahead
            # for C2 continuity, and fits a new polynomial to ref_pos / ref_vel.
            if self._sim_time - self._last_replan > REPLAN_INTERVAL:
                dist_to_ref = float(np.linalg.norm(drone_xyz - ref_pos))
                seg_T = max(LAP_POLY_MIN_T, dist_to_ref / v_target)
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
                # Continue refining the AP while the drone hovers – gate position
                # estimates keep improving as more observations accumulate.
                # If the AP shifts significantly, reset the counter so the drone
                # must re-settle at the updated position before flying through.
                gate_pos_1 = self._gates[self._target_gate]['pos']
                z_new_1    = float(gate_pos_1[2]) if not np.isnan(float(gate_pos_1[2])) else CRUISE_Z
                ap_xy_new  = gate_pos_1[:2] - self._gate_d_hat * APPROACH_DIST
                ap_shift   = float(np.linalg.norm(ap_xy_new - self._gate_ap[:2]))
                if ap_shift > 0.05:
                    self._gate_ap  = np.array([ap_xy_new[0], ap_xy_new[1], z_new_1])
                    self._gate_past = np.array([
                        gate_pos_1[0] + self._gate_d_hat[0] * PAST_DIST,
                        gate_pos_1[1] + self._gate_d_hat[1] * PAST_DIST,
                        z_new_1,
                    ])
                    self._align_count = 0   # must re-settle at the new position
                    ap_xy = self._gate_ap[:2]
                    ap_z  = float(self._gate_ap[2])

                yaw_err = _angle_wrap(self._gate_yaw - yaw)
                yaw_cmd = yaw + float(np.clip(yaw_err, -YAW_RATE_MAX, YAW_RATE_MAX))

                # Require 3-D proximity (XY + Z) and yaw alignment.  Using 3-D
                # distance prevents the drone from counting steps where it is at
                # the right XY position but still at the wrong height.
                dist_3d = float(np.linalg.norm(
                    drone_xyz - np.array([ap_xy[0], ap_xy[1], ap_z])
                ))
                if dist_3d < ALIGN_TOL and abs(yaw_err) < YAW_TOL:
                    self._align_count += 1
                else:
                    self._align_count = 0

                n_obs = len(self._gates[self._target_gate]['observations'])
                if self._align_count >= ALIGN_STEPS and n_obs >= MIN_GATE_OBS:
                    self._ap_phase = 2
                    # Start the through-gate polynomial from the drone's ACTUAL
                    # current position (not the stored gate_ap) so the trajectory
                    # launches cleanly from wherever the drone has settled.
                    self._plan_coeff = _poly5_fit(
                        drone_xyz, np.zeros(3), np.zeros(3),
                        self._gate_past, np.zeros(3), np.zeros(3),
                        T_THROUGH,
                    )
                    self._plan_t0 = self._sim_time
                    self._plan_T  = T_THROUGH
                    print(f"[Nav] gate {self._target_gate + 1} aligned → THROUGH"
                          f"  ({n_obs} obs)")
                elif self._align_count >= ALIGN_STEPS:
                    # Settled but not enough observations yet.
                    # If observations have stopped arriving (gate not in view),
                    # sweep yaw left and right until the gate becomes visible again.
                    if n_obs > self._obs_last_n:
                        self._obs_last_n     = n_obs
                        self._obs_stall_step = 0
                    else:
                        self._obs_stall_step += 1

                    if self._obs_stall_step > OBS_STALL_STEPS:
                        # Override yaw_cmd with a sinusoidal sweep around gate_yaw
                        sweep_yaw = self._gate_yaw + OBS_YAW_AMP * np.sin(
                            self._obs_stall_step * OBS_YAW_SPEED
                        )
                        sweep_err = _angle_wrap(sweep_yaw - yaw)
                        yaw_cmd   = yaw + float(np.clip(sweep_err,
                                                        -YAW_RATE_MAX * 2,
                                                         YAW_RATE_MAX * 2))

                return [float(ap_xy[0]), float(ap_xy[1]), ap_z, yaw_cmd]

            # ── Phase 2: fly through gate via polynomial ───────────────────
            if self._ap_phase == 2:
                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)

                # Gate passed when drone has moved far enough past the gate centre
                gate_pos = self._gates[self._target_gate]['pos']
                proj = float(np.dot(drone_xy - gate_pos[:2], self._gate_d_hat))

                if proj >= PAST_DIST * 0.7 or tau >= self._plan_T:
                    self._gates_order.append(self._target_gate)
                    self._gates_passed.add(self._target_gate)
                    print(f"[Nav] gate {self._target_gate + 1} passed "
                          f"({len(self._gates_passed)}/{NUM_GATES})")
                    if len(self._gates_passed) >= NUM_GATES:
                        _save_results(self._gates, self._gates_order)
                        self._build_lap_spline(drone_xyz)
                        # Project the drone onto the nearest point of the full
                        # spline so lap tracking starts from the current position
                        # rather than flying back to gate 1 first.
                        t_full   = np.linspace(0, self._lap_total_T, 300, endpoint=False)
                        pts_full = self._lap_spline(t_full)
                        best_idx = int(np.argmin(
                            np.linalg.norm(pts_full - drone_xyz, axis=1)
                        ))
                        self._lap_t_progress = float(t_full[best_idx])
                        # Seed the active polynomial toward a point one lookahead
                        # ahead on the spline so the LAP state has a valid target.
                        t_seed  = (self._lap_t_progress + LAP_LOOKAHEAD_T) % self._lap_total_T
                        sp_seed = self._lap_spline(t_seed)
                        sv_seed = self._lap_spline_vel(t_seed)
                        d_seed  = float(np.linalg.norm(drone_xyz - sp_seed))
                        self._plan_coeff  = _poly5_fit(
                            drone_xyz, np.zeros(3), np.zeros(3),
                            sp_seed, sv_seed, np.zeros(3),
                            max(1.0, d_seed / LAP_SPEED),
                        )
                        self._plan_t0     = self._sim_time
                        self._plan_T      = max(1.0, d_seed / LAP_SPEED)
                        self._last_replan = self._sim_time
                        self._nav_state   = self._S_LAP
                        print(f"[Nav] all gates passed → LAP  "
                              f"t_progress={self._lap_t_progress:.2f}")
                        return [drone_xyz[0], drone_xyz[1], CRUISE_Z, yaw]
                    # Build SIDE_STEP_SEGS equally-spaced waypoints 2 m in a direction
                    # 60° to the right of the CCW tangent at the drone's current arena
                    # position.  The tangent (circuit forward direction) is computed from
                    # the drone's radial position relative to ARENA_CENTER, so the
                    # movement is independent of the drone's actual heading.
                    _radial   = drone_xy - ARENA_CENTER
                    _r_len    = float(np.linalg.norm(_radial))
                    _r_hat    = _radial / _r_len if _r_len > 0.1 else np.array([1.0, 0.0])
                    _t_hat    = np.array([-_r_hat[1], _r_hat[0]])   # CCW tangent = circuit forward
                    move_angle = np.arctan2(_t_hat[1], _t_hat[0]) - np.radians(60)
                    move_dir   = np.array([np.cos(move_angle), np.sin(move_angle)])
                    _xy_lo = ARENA_MARGIN
                    _xy_hi = 8.0 - ARENA_MARGIN
                    self._side_step_wps = [
                        np.array([
                            float(np.clip(drone_xyz[0] + move_dir[0] * SIDE_STEP_DIST * k / SIDE_STEP_SEGS,
                                          _xy_lo, _xy_hi)),
                            float(np.clip(drone_xyz[1] + move_dir[1] * SIDE_STEP_DIST * k / SIDE_STEP_SEGS,
                                          _xy_lo, _xy_hi)),
                            float(drone_xyz[2]),
                        ])
                        for k in range(1, SIDE_STEP_SEGS + 1)
                    ]
                    self._side_step_idx = 0
                    self._nav_state = self._S_SIDE_STEP
                    print("[Nav] → SIDE_STEP")
                    # fall through to SIDE_STEP
                else:
                    return [float(sp[0]), float(sp[1]), float(sp[2]), self._gate_yaw]

        # ── SIDE_STEP: smooth C2-continuous movement through waypoints ────
        if self._nav_state == self._S_SIDE_STEP:
            target = self._side_step_wps[self._side_step_idx]

            # Advance to the next waypoint when close enough
            if float(np.linalg.norm(drone_xy - target[:2])) < SIDE_STEP_TOL:
                self._side_step_idx += 1
                if self._side_step_idx >= len(self._side_step_wps):
                    self._nav_state   = self._S_SEARCH
                    self._search_yaw  = yaw
                    self._search_step = 0
                    print("[Nav] side-step complete → SEARCH")
                    # fall through to SEARCH
                else:
                    target = self._side_step_wps[self._side_step_idx]

            if self._nav_state == self._S_SIDE_STEP:
                # Same continuous replanning as approach phase 0: fire every
                # REPLAN_INTERVAL, sample T_REPLAN_BUF ahead on the current
                # polynomial for C2 continuity, replan toward current waypoint.
                if self._sim_time - self._last_replan > REPLAN_INTERVAL:
                    dist = float(np.linalg.norm(drone_xyz - target))
                    self._replan_to(target, duration=max(SIDE_STEP_DUR, dist / 1.5))

                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)
                return [float(sp[0]), float(sp[1]), float(sp[2]), yaw]

        # ── SEARCH: rotate in place until a gate is detected ─────────────
        if self._nav_state == self._S_SEARCH:
            next_gate = self._nearest_unvisited(drone_xy)
            if next_gate is not None:
                self._target_gate = next_gate
                self._plan_to_gate(next_gate, drone_xyz)
                self._nav_state = self._S_APPROACH
                print(f"[Nav] target gate {next_gate + 1} → APPROACH")
                tau = self._sim_time - self._plan_t0
                sp, _, _ = _poly5_eval(self._plan_coeff, tau, self._plan_T)
                return [float(sp[0]), float(sp[1]), float(sp[2]), self._gate_yaw]

            # No gate detected yet – rotate in place with a gentle Z oscillation.
            self._search_yaw  += SEARCH_YAW_RATE
            self._search_step += 1
            z_sp = CRUISE_Z + SEARCH_Z_AMP * np.sin(self._search_step * SEARCH_Z_RATE)
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


def _save_results(gates, gates_order):
    """Write estimated vs ground-truth gate positions to _RESULTS_FILE.

    Gates are listed in the order they were passed through (gates_order),
    so Gate 1 in the output is the first gate the drone flew through, etc.
    Truth matching uses the detection index (gate_idx) which corresponds to
    the Webots GATE0–GATE4 definition order.
    """
    import json

    truth = []
    if os.path.exists(_TRUTH_FILE):
        with open(_TRUTH_FILE) as f:
            truth = json.load(f)

    lines = ['Gate  |   Estimated (x, y, z)          |   Ground truth (x, y, z)       |  Error (m)  |  Obs',
             '-' * 100]
    for pass_num, gate_idx in enumerate(gates_order):
        g       = gates[gate_idx]
        est     = g['pos']
        est_str = f'({est[0]:6.3f}, {est[1]:6.3f}, {est[2]:6.3f})'
        if gate_idx < len(truth):
            gt      = truth[gate_idx]
            err     = float(np.linalg.norm(est - np.array([gt['x'], gt['y'], gt['z']])))
            gt_str  = f'({gt["x"]:6.3f}, {gt["y"]:6.3f}, {gt["z"]:6.3f})'
            err_str = f'{err:.3f}'
        else:
            gt_str  = 'N/A'
            err_str = 'N/A'
        n_obs = len(g['observations'])
        lines.append(f'  {pass_num+1}   |  {est_str}  |  {gt_str}  |  {err_str:>9}  |  {n_obs} obs')

    with open(_RESULTS_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'[Results] {_RESULTS_FILE}')
    for line in lines:
        print(line)


_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
