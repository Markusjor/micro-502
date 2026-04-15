# Aerial Robotics 
Exercises and the major software project using a Crazyflie implemented in Webots for the MICRO-502 Aerial Robotics course.

**Documentation:** https://micro-502.readthedocs.io

---

## Tuning reference

All constants are at the top of `controllers/main/assignment/my_assignment.py`.

### Seed / reproducibility

| Constant | Default | Effect |
|---|---|---|
| `rand_seed` *(main.py)* | `None` | Set to an integer to replay a specific run; `None` = random each time |

### Gate detection

| Constant | Default | Effect |
|---|---|---|
| `_SAVE_DEDUP_RADIUS` | 1.2 m | Detections within this radius are merged into the same gate. Increase if the same gate is registered twice; decrease if two nearby gates are incorrectly merged. |
| `MIN_GATE_OBS` | 200 | Minimum observations accumulated before the drone commits to flying through a gate. Lower = faster but less accurate gate position estimate. |
| `MIN_AREA` | 200 px² | Minimum contour area to be considered a gate panel. Raise to reject noise, lower to detect small/distant gates. |

### Lap 1 approach & alignment

| Constant | Default | Effect |
|---|---|---|
| `APPROACH_DIST` | 0.4 m | How far in front of the gate the approach point is placed. |
| `PAST_DIST` | 1.2 m | Distance past the gate centre before the gate is counted as passed. |
| `ALIGN_STEPS` | 80 steps | Consecutive on-target control steps required before committing to fly through. Lower = less dwell time, higher miss risk. |
| `ALIGN_TOL` | 0.10 m | Position tolerance during alignment phase. |
| `YAW_TOL` | 0.12 rad | Yaw tolerance during alignment phase. |
| `OBS_STALL_STEPS` | 25 steps | Steps without a new observation before the yaw sweep kicks in. |
| `OBS_YAW_AMP` | 0.35 rad | Amplitude of the yaw sweep used to gather more observations when stalled. |

### Side-step (after each gate on lap 1)

| Constant | Default | Effect |
|---|---|---|
| `SIDE_STEP_DIST` | 2.0 m | Total distance moved sideways after passing a gate. |
| `SIDE_STEP_SEGS` | 3 | Number of waypoints the side-step is split into. |
| `SIDE_STEP_DUR` | 1.2 s | Polynomial duration per waypoint segment — lower = faster side-step. |
| `SIDE_STEP_TOL` | 0.20 m | Arrival radius per waypoint. |

### Laps 2 & 3 (spline racing)

| Constant | Default | Effect |
|---|---|---|
| `LAP_SPEED` | 6.0 m/s | Target speed. Polynomial duration = `dist / LAP_SPEED`. Higher = more aggressive. |
| `LAP_LOOKAHEAD_T` | 0.6 s | How far ahead on the spline the tracking target is placed. Lower = follows path more faithfully; higher = cuts corners and increases speed. |
| `LAP_SEG_T` | 2.0 s | Spline parameterisation: time assigned per gate-to-gate segment. Affects spline shape only, not flight speed. |
| `LAP_POLY_MIN_T` | 0.15 s | Floor on polynomial duration — prevents near-zero-duration replans. |

### Replanning (applies to all states)

| Constant | Default | Effect |
|---|---|---|
| `REPLAN_INTERVAL` | 0.10 s | Minimum time between successive replans. Lower = more responsive tracking. |
| `T_REPLAN_BUF` | 0.2 s | Time ahead on the current polynomial sampled as the C2-continuity seed for the next plan. Lower = less lag before the new plan takes effect. |