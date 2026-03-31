import os
import json
import numpy as np
import cv2

# The available ground truth state measurements can be accessed by calling sensor_data[item]. All values of "item" are provided as defined in main.py within the function read_sensors.
# The "item" values that you may later retrieve for the hardware project are:
# "x_global": Global X position
# "y_global": Global Y position
# "z_global": Global Z position
# 'v_x": Global X velocity
# "v_y": Global Y velocity
# "v_z": Global Z velocity
# "ax_global": Global X acceleration
# "ay_global": Global Y acceleration
# "az_global": Global Z acceleration (With gravtiational acceleration subtracted)
# "roll": Roll angle (rad)
# "pitch": Pitch angle (rad)
# "yaw": Yaw angle (rad)
# "q_x": X Quaternion value
# "q_y": Y Quaternion value
# "q_z": Z Quaternion value
# "q_w": W Quaternion value

# A link to further information on how to access the sensor data on the Crazyflie hardware for the hardware practical can be found here: https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/api/logs/#stateestimate

# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------
_PROJECT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GATE_MAP_FILE = os.path.join(_PROJECT_DIR, 'gate_map.json')
GATE_IMG_DIR  = os.path.join(_PROJECT_DIR, 'gate_detections')
os.makedirs(GATE_IMG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Camera intrinsics  (from crazyflie_world_assignment.wbt)
# ---------------------------------------------------------------------------
IMG_W = 300
IMG_H = 300
FOV   = 1.5
F_PX  = (IMG_W / 2) / np.tan(FOV / 2)    # focal length ≈ 161 px

# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------
GATE_PANEL_H = 0.40   # m – fixed pink panel height

# ---------------------------------------------------------------------------
# HSV thresholds – pink/magenta wraps around hue 0/179
# ---------------------------------------------------------------------------
# S>=50 captures the transparent (0.2) emissive panel; V>=30 handles distance.
# The camera frame is pure magenta but is stripped by BORDER before detection.
HSV_LO1 = np.array([140,  50,  30])
HSV_HI1 = np.array([179, 255, 255])
HSV_LO2 = np.array([  0,  50,  30])
HSV_HI2 = np.array([ 10, 255, 255])
BORDER       = 8     # px – strip camera frame before detection
MIN_AREA     = 250   # px² – rejects small noise/sign blobs
MIN_SOLIDITY = 0.55  # slightly relaxed so semi-transparent panels still pass
MAX_ASPECT   = 1.6   # gate panel is ~square (0.4×0.4 m); rejects wide signs
MIN_PIXEL_H  = 8     # px – minimum apparent panel height for depth estimation

# ---------------------------------------------------------------------------
# VIO / mapping parameters
# ---------------------------------------------------------------------------
NUM_GATES    = 5
DEDUP_RADIUS = 1.2   # m – XY radius for same-gate matching

MIN_BASELINE  = 0.4   # m – minimum drone displacement needed to triangulate
MIN_OBS       = 2     # minimum observations before a gate can be confirmed
MAX_COND      = 500   # max condition number of triangulation matrix A
MAX_RESIDUAL  = 0.50  # m – max per-ray residual for confirmation
BEARING_TOL   = 0.30  # rad – angular match tolerance for bearing-only match

# Depth-mean fallback: fires after this many obs from a stationary position.
# Low value means gates are confirmed quickly from depth-from-panel-size alone,
# giving an approximate position that gets refined once the drone moves.
FALLBACK_OBS           = 3
MAX_RAYS_PER_CANDIDATE = 60
MAX_CANDIDATES         = 25

ARENA_XMIN, ARENA_XMAX = 0.0, 9.0
ARENA_YMIN, ARENA_YMAX = 0.0, 9.0
ARENA_ZMIN, ARENA_ZMAX = 0.4, 2.5

# ---------------------------------------------------------------------------
# Navigation parameters
# ---------------------------------------------------------------------------
CRUISE_Z          = 1.0   # m – default flight altitude
APPROACH_DIST     = 1.5   # m – approach wp distance in front of gate
PAST_DIST         = 1.2   # m – how far past the gate centre before switching to SEARCH
TRAJ_STEP_THRESH  = 0.35  # m – advance to next trajectory setpoint when within this distance
T_SEG             = 2.5   # s – time budget per waypoint-to-waypoint segment
DISC_STEPS        = 20    # trajectory points per segment (matches ex3)

# Exploration waypoints: sweep the arena so every gate is visible from at
# least two positions with sufficient baseline.  The drone starts at ~(1, 4).
EXPLORE_WAYPOINTS = [
    (3.0, 2.0, 1.2),
    (6.0, 2.5, 1.2),
    (7.0, 4.0, 1.2),
    (5.5, 6.5, 1.2),
    (2.0, 6.5, 1.2),
    (1.0, 4.0, 1.2),
]
EXPLORE_WP_THRESH = 0.4
SEARCH_YAW_RATE   = 0.04   # rad per control step while scanning for next gate


# ---------------------------------------------------------------------------
# Gate corner detection
# ---------------------------------------------------------------------------
def detect_gate_corners(bgr_image):
    hsv  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, HSV_LO1, HSV_HI1),
                          cv2.inRange(hsv, HSV_LO2, HSV_HI2))
    # Strip camera frame (pure magenta border) before morphology so it doesn't
    # bleed inward and dominate contour finding.
    mask[:BORDER, :]  = 0
    mask[-BORDER:, :] = 0
    mask[:, :BORDER]  = 0
    mask[:, -BORDER:] = 0
    # 7×7 close fills interior holes from the semi-transparent panel without
    # expanding small noise fragments into false positives.
    k = np.ones((7, 7), np.uint8)
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
        w, h = rect[1]
        if min(w, h) < 1 or max(w, h) / min(w, h) > MAX_ASPECT:
            continue
        # Always use the 4 corners of the minimum bounding rectangle.
        # approxPolyDP on a semi-transparent panel rarely gives exactly 4
        # vertices; boxPoints always does and is sufficient for pixel_h.
        corners = cv2.boxPoints(rect).astype(np.float32)
        results.append(corners)
    return results


# ---------------------------------------------------------------------------
# Triangulation helpers
# ---------------------------------------------------------------------------
def triangulate_rays(origins, directions, depths=None):
    """
    Weighted least-squares ray intersection.

    Weight = 1/depth² so closer observations (lower depth estimation noise)
    contribute more than distant ones.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for i, (p, d) in enumerate(zip(origins, directions)):
        d  = d / np.linalg.norm(d)
        w  = 1.0 / max(depths[i], 0.3) ** 2 if depths is not None else 1.0
        M  = np.eye(3) - np.outer(d, d)
        A += w * M
        b += w * (M @ p)
    try:
        cond = float(np.linalg.cond(A))
        pos  = np.linalg.solve(A, b)
        return pos, cond
    except np.linalg.LinAlgError:
        return None, float('inf')


def ray_residual(origins, directions, pos):
    """Mean perpendicular distance from pos to all rays."""
    total = 0.0
    for p, d in zip(origins, directions):
        d     = d / np.linalg.norm(d)
        diff  = pos - p
        perp  = diff - np.dot(diff, d) * d
        total += float(np.linalg.norm(perp))
    return total / len(origins)


# ---------------------------------------------------------------------------
# Main assignment class
# ---------------------------------------------------------------------------
class MyAssignment:

    # Navigation states
    _S_EXPLORE  = 'explore'
    _S_APPROACH = 'approach'
    _S_SEARCH   = 'search'
    _S_DONE     = 'done'

    def __init__(self):
        self.candidates  = []   # all gate candidates (confirmed + unconfirmed)
        self._explore_wp = 0    # index into EXPLORE_WAYPOINTS

        # State machine
        self._nav_state       = self._S_EXPLORE
        self._gates_passed    = set()   # gate_idx values already flown through

        # Trajectory for current gate
        self._traj            = None    # (N, 3) array of xyz setpoints
        self._traj_idx        = 0
        self._target_gate     = None    # gate_idx currently being approached
        self._centre_traj_idx = None    # traj index corresponding to gate centre

        # Search (post-gate rotation scan)
        self._search_yaw      = 0.0

    # -----------------------------------------------------------------
    # Confirmed gate map
    # -----------------------------------------------------------------
    @property
    def gate_map(self):
        return [c['pos'] for c in self.candidates if c['confirmed']]

    # -----------------------------------------------------------------
    # Observation extraction
    # -----------------------------------------------------------------
    def _corners_to_obs(self, corners, sensor_data):
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))

        sorted_y = corners[np.argsort(corners[:, 1])]
        pixel_h  = float(np.mean(sorted_y[2:, 1]) - np.mean(sorted_y[:2, 1]))
        if pixel_h < MIN_PIXEL_H:
            return None

        # depth_fwd is the forward (Z_camera / along optical axis) distance.
        depth_fwd = float(np.clip(F_PX * GATE_PANEL_H / pixel_h, 0.3, 12.0))

        x_n = (cx - IMG_W / 2) / F_PX
        y_n = (cy - IMG_H / 2) / F_PX

        # Ray in camera/body frame: forward=X, right=-Y, up=Z
        # Pixel right (+x_n) → body -Y; pixel down (+y_n) → body -Z
        d_body = np.array([1.0, -x_n, -y_n])

        # Correct for off-centre projection: depth_fwd is the component along
        # the optical axis (body X), not the 3-D ray distance.
        # 3-D distance = depth_fwd * |d_body|  (since d_body[0] = 1)
        ray_scale = float(np.linalg.norm(d_body))   # sqrt(1 + x_n² + y_n²)
        depth_3d  = depth_fwd * ray_scale

        # Full rotation R = Rz(yaw) * Ry(pitch) * Rx(roll)
        roll  = sensor_data['roll']
        pitch = sensor_data['pitch']
        yaw   = sensor_data['yaw']
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)
        R = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,    cp*sr,             cp*cr            ]
        ])
        d = R @ d_body
        d /= np.linalg.norm(d)

        origin = np.array([sensor_data['x_global'],
                           sensor_data['y_global'],
                           sensor_data['z_global']])
        return origin, d, depth_3d

    # -----------------------------------------------------------------
    # Candidate helpers
    # -----------------------------------------------------------------
    def _new_candidate(self, origin, direction, depth_est, frame, corners):
        pos_est = origin + depth_est * direction
        return {
            'origins':      [origin.copy()],
            'directions':   [direction.copy()],
            'depths':       [depth_est],
            'pos':          pos_est.copy(),
            'best_residual': float('inf'),   # best residual seen so far
            'triangulated': False,
            'confirmed':    False,
            'gate_idx':     None,
            'last_frame':   frame.copy(),
            'last_corners': corners.copy(),
        }

    def _match(self, origin, direction, depth_est):
        pos_est = origin + depth_est * direction
        for i, c in enumerate(self.candidates):
            if np.linalg.norm(pos_est[:2] - c['pos'][:2]) < DEDUP_RADIUS:
                return i
            if not c['confirmed']:
                to_cand = c['pos'][:2] - origin[:2]
                dist    = np.linalg.norm(to_cand)
                if dist > 0.5:
                    angle_cand = np.arctan2(to_cand[1], to_cand[0])
                    angle_obs  = np.arctan2(direction[1], direction[0])
                    diff = abs((angle_cand - angle_obs + np.pi) % (2 * np.pi) - np.pi)
                    if diff < BEARING_TOL:
                        return i
        return -1

    def _add_ray(self, cand, origin, direction, depth_est, frame, corners):
        cand['last_frame']   = frame.copy()
        cand['last_corners'] = corners.copy()

        # Always collect depth estimates – these power the stationary fallback.
        cand['depths'].append(depth_est)

        # Only store a new ray origin when the drone has moved enough.
        stationary = any(np.linalg.norm(origin[:2] - prev[:2]) < 0.05
                         for prev in cand['origins'])
        if not stationary:
            cand['origins'].append(origin.copy())
            cand['directions'].append(direction.copy())
            if len(cand['origins']) > MAX_RAYS_PER_CANDIDATE:
                cand['origins'].pop(0)
                cand['directions'].pop(0)

        if len(cand['depths']) > MAX_RAYS_PER_CANDIDATE:
            cand['depths'].pop(0)

        # Primary: triangulation from spatially diverse rays
        if len(cand['origins']) >= 2:
            origs    = np.array(cand['origins'])
            baseline = float(np.max(np.linalg.norm(
                origs[:, :2] - origs[:1, :2], axis=1)))
            if baseline >= MIN_BASELINE:
                pos, cond = triangulate_rays(cand['origins'], cand['directions'],
                                             cand['depths'][:len(cand['origins'])])
                if pos is not None and cond <= MAX_COND:
                    new_res = ray_residual(cand['origins'], cand['directions'], pos)
                    if new_res < cand['best_residual']:
                        cand['pos']           = pos
                        cand['best_residual'] = new_res
                        cand['triangulated']  = True
                        return True

        # Fallback: weighted depth mean from repeated observations at same position.
        # Uses only stored rays (may be just 1) paired with all depth estimates.
        if len(cand['depths']) >= FALLBACK_OBS:
            depths  = np.array(cand['depths'])
            weights = 1.0 / np.maximum(depths, 0.3)
            # Repeat the single/first origin+direction for all depth samples
            o0, d0  = cand['origins'][0], cand['directions'][0]
            pts     = np.array([o0 + d0 * z for z in depths])
            pos     = np.average(pts, axis=0, weights=weights)
            new_res = ray_residual(cand['origins'], cand['directions'], pos)
            if new_res < cand['best_residual']:
                cand['pos']           = pos
                cand['best_residual'] = new_res
                cand['triangulated']  = True
                return True

        return False

    def _try_confirm(self, cand):
        if not cand['triangulated']:
            return
        # For the depth-mean fallback (stationary drone, 1 origin) require
        # enough depth samples instead of spatial observations.
        if len(cand['origins']) < MIN_OBS and len(cand['depths']) < FALLBACK_OBS:
            return

        pos     = cand['pos']
        residual = cand['best_residual']

        if not (ARENA_XMIN < pos[0] < ARENA_XMAX and
                ARENA_YMIN < pos[1] < ARENA_YMAX and
                ARENA_ZMIN < pos[2] < ARENA_ZMAX):
            return
        if residual > MAX_RESIDUAL:
            return

        if cand['confirmed']:
            # Position was already updated only if residual improved in _add_ray;
            # just persist and log the new best.
            print(f"[GateMap] Gate {cand['gate_idx']} refined  "
                  f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f}  "
                  f"residual={residual:.3f}m  n={len(cand['origins'])}")
            _save_gate_map(self.gate_map)
            _save_detection_image(cand['gate_idx'], cand['last_frame'],
                                  cand['last_corners'], pos)
            return

        # First confirmation – check not a duplicate
        for c in self.candidates:
            if c is not cand and c['confirmed']:
                if np.linalg.norm(pos[:2] - c['pos'][:2]) < DEDUP_RADIUS:
                    return

        n_confirmed = sum(1 for c in self.candidates if c['confirmed'])
        if n_confirmed >= NUM_GATES:
            return

        gate_idx          = n_confirmed + 1
        cand['confirmed'] = True
        cand['gate_idx']  = gate_idx
        print(f"[GateMap] Gate {gate_idx}/{NUM_GATES} confirmed  "
              f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f}  "
              f"residual={residual:.3f}m  n={len(cand['origins'])}")
        if gate_idx == NUM_GATES:
            print("[GateMap] All gates found!")
        _save_gate_map(self.gate_map)
        _save_detection_image(gate_idx, cand['last_frame'], cand['last_corners'], pos)

    # -----------------------------------------------------------------
    # Observation pipeline
    # -----------------------------------------------------------------
    def _process_observation(self, origin, direction, depth_est, frame, corners):
        idx = self._match(origin, direction, depth_est)

        if idx < 0:
            if len(self.candidates) >= MAX_CANDIDATES:
                return
            self.candidates.append(
                self._new_candidate(origin, direction, depth_est, frame, corners))
            return

        updated = self._add_ray(self.candidates[idx], origin, direction,
                                depth_est, frame, corners)
        if updated:
            self._try_confirm(self.candidates[idx])

    # -----------------------------------------------------------------
    # Navigation helpers
    # -----------------------------------------------------------------
    def _gate_candidate(self, gate_idx):
        for c in self.candidates:
            if c['gate_idx'] == gate_idx:
                return c
        return None

    def _nearest_unvisited(self, drone_xy):
        """Return gate_idx of nearest confirmed unvisited gate, or None."""
        best_dist, best_idx = float('inf'), None
        for c in self.candidates:
            if c['confirmed'] and c['gate_idx'] not in self._gates_passed:
                d = float(np.linalg.norm(c['pos'][:2] - drone_xy))
                if d < best_dist:
                    best_dist, best_idx = d, c['gate_idx']
        return best_idx

    def _plan_to_gate(self, gate_idx, drone_xyz):
        """Build a 4-waypoint min-jerk trajectory: start → approach → centre → past."""
        cand     = self._gate_candidate(gate_idx)
        gate_pos = cand['pos']
        drone_xy = drone_xyz[:2]
        to_gate  = gate_pos[:2] - drone_xy
        dist     = float(np.linalg.norm(to_gate))
        fwd      = to_gate / dist if dist > 0.01 else np.array([1.0, 0.0])
        ap       = gate_pos[:2] - fwd * APPROACH_DIST
        past     = gate_pos[:2] + fwd * PAST_DIST
        z        = float(gate_pos[2])

        waypoints = [
            drone_xyz.tolist(),
            [float(ap[0]), float(ap[1]), z],
            [float(gate_pos[0]), float(gate_pos[1]), z],
            [float(past[0]),     float(past[1]),     z],
        ]
        traj = self._min_jerk_trajectory(waypoints)
        self._traj            = traj
        self._traj_idx        = 0
        self._target_gate     = gate_idx
        self._centre_traj_idx = len(traj) - 1   # drone must reach past-gate point
        print(f"[Nav] Trajectory to gate {gate_idx}: {len(traj)} pts, "
              f"gate @ ({gate_pos[0]:.2f}, {gate_pos[1]:.2f}, {gate_pos[2]:.2f})")

    # -----------------------------------------------------------------
    # Minimum-jerk polynomial planner  (same method as ex3)
    # -----------------------------------------------------------------
    def _poly_matrix(self, t):
        """5×6 constraint matrix at time t (positions, vel, acc, jerk, snap)."""
        return np.array([
            [t**5,    t**4,    t**3,   t**2,  t,  1],
            [5*t**4,  4*t**3,  3*t**2, 2*t,   1,  0],
            [20*t**3, 12*t**2, 6*t,    2,     0,  0],
            [60*t**2, 24*t,    6,      0,     0,  0],
            [120*t,   24,      0,      0,     0,  0],
        ])

    def _min_jerk_trajectory(self, waypoints):
        """
        Compute minimum-jerk 5th-order polynomial trajectory through waypoints.
        Returns (N×3 setpoint array, list of segment boundary indices).
        Directly mirrors ex3_motion_planner.compute_poly_coefficients /
        poly_setpoint_extraction.
        """
        m    = len(waypoints)
        segs = m - 1
        times     = np.linspace(0, T_SEG * segs, m)
        seg_times = np.diff(times)
        A_0       = self._poly_matrix(0)

        poly_coeffs = np.zeros((6 * segs, 3))

        for dim in range(3):
            A = np.zeros((6 * segs, 6 * segs))
            b = np.zeros(6 * segs)
            pos = np.array([p[dim] for p in waypoints])
            row = 0
            for i in range(segs):
                A_f = self._poly_matrix(seg_times[i])
                if i == 0:
                    A[row, 0:6] = A_0[0]; b[row] = pos[0]; row += 1   # pos(0)
                    A[row, 0:6] = A_f[0]; b[row] = pos[1]; row += 1   # pos(T)
                    A[row, 0:6] = A_0[1]; b[row] = 0;      row += 1   # vel(0)=0
                    A[row, 0:6] = A_0[2]; b[row] = 0;      row += 1   # acc(0)=0
                    if segs > 1:
                        A[row:row+4, 0:6]   = A_f[1:]
                        A[row:row+4, 6:12]  = -A_0[1:]
                        b[row:row+4]        = 0; row += 4             # continuity
                    else:
                        A[row, 0:6] = A_f[1]; b[row] = 0; row += 1   # vel(T)=0
                        A[row, 0:6] = A_f[2]; b[row] = 0; row += 1   # acc(T)=0
                elif i < segs - 1:
                    s, e = i*6, (i+1)*6
                    A[row, s:e] = A_0[0]; b[row] = pos[i];   row += 1
                    A[row, s:e] = A_f[0]; b[row] = pos[i+1]; row += 1
                    A[row:row+4, s:e]     = A_f[1:]
                    A[row:row+4, e:e+6]   = -A_0[1:]
                    b[row:row+4] = 0; row += 4
                else:
                    s, e = i*6, (i+1)*6
                    A[row, s:e] = A_0[0]; b[row] = pos[i];   row += 1
                    A[row, s:e] = A_f[0]; b[row] = pos[i+1]; row += 1
                    A[row, s:e] = A_f[1]; b[row] = 0;        row += 1  # vel(T)=0
                    A[row, s:e] = A_f[2]; b[row] = 0;        row += 1  # acc(T)=0

            poly_coeffs[:, dim] = np.linalg.solve(A, b)

        # Extract fine setpoints
        t_fine = np.linspace(0, times[-1], DISC_STEPS * segs)
        pts    = []
        seg_boundaries = []   # t_fine indices at start of each segment
        for i, t in enumerate(t_fine):
            seg = min(max(np.searchsorted(times, t) - 1, 0), segs - 1)
            t_r = t - times[seg]
            row = self._poly_matrix(t_r)[0]
            x   = np.dot(row, poly_coeffs[seg*6:(seg+1)*6, 0])
            y   = np.dot(row, poly_coeffs[seg*6:(seg+1)*6, 1])
            z   = np.dot(row, poly_coeffs[seg*6:(seg+1)*6, 2])
            pts.append([x, y, z])
        return np.array(pts)

    def _navigate(self, sensor_data):
        x   = sensor_data['x_global']
        y   = sensor_data['y_global']
        z   = sensor_data['z_global']
        drone_xy  = np.array([x, y])
        drone_xyz = np.array([x, y, z])

        # ----------------------------------------------------------------
        # DONE
        # ----------------------------------------------------------------
        if self._nav_state == self._S_DONE:
            return [x, y, CRUISE_Z, sensor_data['yaw']]

        # ----------------------------------------------------------------
        # APPROACH – check gate passage before moving the setpoint forward
        # ----------------------------------------------------------------
        if self._nav_state == self._S_APPROACH and self._target_gate is not None:
            if (self._centre_traj_idx is not None and
                    self._traj_idx >= self._centre_traj_idx):
                self._gates_passed.add(self._target_gate)
                print(f"[Nav] Gate {self._target_gate} passed  "
                      f"({len(self._gates_passed)}/{NUM_GATES})")
                if len(self._gates_passed) >= NUM_GATES:
                    self._nav_state = self._S_DONE
                    print("[Nav] All gates passed!")
                    return [x, y, CRUISE_Z, sensor_data['yaw']]
                # Hover here and rotate to search for the next gate
                self._nav_state   = self._S_SEARCH
                self._search_yaw  = float(sensor_data['yaw'])
                self._traj        = None
                self._target_gate = None

        # ----------------------------------------------------------------
        # SEARCH – hold position, rotate until next gate is confirmed
        # ----------------------------------------------------------------
        if self._nav_state == self._S_SEARCH:
            next_gate = self._nearest_unvisited(drone_xy)
            if next_gate is not None:
                self._plan_to_gate(next_gate, drone_xyz)
                self._nav_state = self._S_APPROACH
                # Fall through to APPROACH handler below
            else:
                self._search_yaw += SEARCH_YAW_RATE
                return [x, y, float(z), self._search_yaw]

        # ----------------------------------------------------------------
        # EXPLORE – sweep arena waypoints until a gate is confirmed
        # ----------------------------------------------------------------
        if self._nav_state == self._S_EXPLORE:
            next_gate = self._nearest_unvisited(drone_xy)
            if next_gate is not None:
                self._plan_to_gate(next_gate, drone_xyz)
                self._nav_state = self._S_APPROACH
                # Fall through to APPROACH handler below
            else:
                wp    = EXPLORE_WAYPOINTS[self._explore_wp % len(EXPLORE_WAYPOINTS)]
                wp_xy = np.array(wp[:2])
                if float(np.linalg.norm(wp_xy - drone_xy)) < EXPLORE_WP_THRESH:
                    self._explore_wp += 1
                    wp    = EXPLORE_WAYPOINTS[self._explore_wp % len(EXPLORE_WAYPOINTS)]
                    wp_xy = np.array(wp[:2])
                to_wp = wp_xy - drone_xy
                tyaw  = float(np.arctan2(to_wp[1], to_wp[0]))
                return [float(wp_xy[0]), float(wp_xy[1]), float(wp[2]), tyaw]

        # ----------------------------------------------------------------
        # APPROACH – follow min-jerk trajectory setpoints
        # ----------------------------------------------------------------
        if self._nav_state == self._S_APPROACH and self._traj is not None:
            idx = min(self._traj_idx, len(self._traj) - 1)
            sp  = self._traj[idx]
            if float(np.linalg.norm(drone_xyz - sp)) < TRAJ_STEP_THRESH:
                if self._traj_idx < len(self._traj) - 1:
                    self._traj_idx += 1
                    sp = self._traj[self._traj_idx]
            to_sp = sp[:2] - drone_xy
            dist  = float(np.linalg.norm(to_sp))
            tyaw  = (float(np.arctan2(to_sp[1], to_sp[0]))
                     if dist > 0.1 else sensor_data['yaw'])
            return [float(sp[0]), float(sp[1]), float(sp[2]), tyaw]

        # Fallback
        return [x, y, CRUISE_Z, sensor_data['yaw']]

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def compute_command(self, sensor_data, camera_data, _dt):
        if sensor_data['z_global'] < 0.49:
            return [sensor_data['x_global'], sensor_data['y_global'],
                    1.0, sensor_data['yaw']]

        bgr        = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        detections = detect_gate_corners(bgr)

        for corners in detections:
            obs = self._corners_to_obs(corners, sensor_data)
            if obs is not None:
                self._process_observation(*obs, bgr, corners)

        return self._navigate(sensor_data)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _save_detection_image(gate_idx, bgr, corners, world_pos):
    out = bgr.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    for pt in corners:
        cv2.circle(out, tuple(pt.astype(int)), 4, (0, 0, 255), -1)
    label = f"Gate {gate_idx}  ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
    cv2.putText(out, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1, cv2.LINE_AA)
    path = os.path.join(GATE_IMG_DIR, f'gate_{gate_idx}.png')
    cv2.imwrite(path, out)
    print(f"[GateMap] Saved detection image → {path}")


def _save_gate_map(gate_map):
    data = [{'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
            for p in gate_map]
    with open(GATE_MAP_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_controller = MyAssignment()

def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)

def print_gate_map():
    gate_map = _controller.gate_map
    print("\n" + "=" * 50)
    print(f"GATE MAP SUMMARY  ({len(gate_map)}/{NUM_GATES} gates discovered)")
    print("=" * 50)
    if not gate_map:
        print("  No gates were detected during this run.")
    for i, pos in enumerate(gate_map):
        print(f"  Gate {i + 1}: x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
    print(f"  Saved to: {GATE_MAP_FILE}")
    print("=" * 50)
    _save_gate_map(gate_map)
