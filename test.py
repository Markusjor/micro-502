"""
Gate corner detection for the micro-502 assignment.

Pipeline (fast enough for real-time use):
  1. Convert BGR → HSV
  2. Threshold for the pink/magenta panel colour
  3. Morphological clean-up (small kernel)
  4. Find external contours
  5. For each large-enough contour:
       a. approxPolyDP  → ideally gives 4 corners directly
       b. fall back to minAreaRect corners if poly has ≠ 4 vertices
  6. Return list of (4×2) corner arrays (one per detected gate panel)

Run this file directly to test on test1.png / test2.png with timing info.
"""

import cv2
import numpy as np
import time
import os

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
# HSV range for the magenta/pink panels
# Magenta wraps around the 0/179 boundary in OpenCV's 0-179 hue scale
HSV_LO1 = np.array([140,  50,  30])
HSV_HI1 = np.array([179, 255, 255])
HSV_LO2 = np.array([  0,  50,  30])
HSV_HI2 = np.array([ 10, 255, 255])

BORDER     = 8      # px – strip camera frame before detection
MIN_AREA   = 50     # px²
MIN_SOLIDITY = 0.55
MAX_ASPECT   = 2.5
POLY_EPS   = 0.08


def detect_gate_corners(bgr_image):
    """
    Detect pink gate panel corners in a BGR image.

    Returns
    -------
    list of np.ndarray, shape (4, 2), dtype int
        Each array holds the four (x, y) pixel corners of one detected gate,
        ordered by minAreaRect (bottom-left, top-left, top-right, bottom-right
        in the rotated-rect convention).
    """
    hsv  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(cv2.inRange(hsv, HSV_LO1, HSV_HI1),
                          cv2.inRange(hsv, HSV_LO2, HSV_HI2))

    # Strip camera frame before morphology
    mask[:BORDER, :]  = 0
    mask[-BORDER:, :] = 0
    mask[:, :BORDER]  = 0
    mask[:, -BORDER:] = 0

    k    = np.ones((5, 5), np.uint8)
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
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, POLY_EPS * peri, True)
        corners = approx.reshape(4, 2) if len(approx) == 4 else cv2.boxPoints(rect).astype(int)
        results.append(corners)

    return results


def draw_corners(image, gates, colour=(0, 255, 0), thickness=2, dot_r=5):
    """Draw detected gate corners and edges on a copy of *image*."""
    out = image.copy()
    for corners in gates:
        # Draw the 4 edges of the quadrilateral
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(out, pt1, pt2, colour, thickness)
        # Draw each corner as a filled circle
        for pt in corners:
            cv2.circle(out, tuple(pt), dot_r, (0, 0, 255), -1)
    return out


# ---------------------------------------------------------------------------
# Stand-alone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images = [
        os.path.join(script_dir, 'test1.png'),
        os.path.join(script_dir, 'test2.png'),
    ]

    for img_path in images:
        if not os.path.exists(img_path):
            print(f'[SKIP] {img_path} not found')
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f'[SKIP] could not read {img_path}')
            continue

        # --- Timing: run 100 iterations to get a stable estimate ---
        N = 100
        t0 = time.perf_counter()
        for _ in range(N):
            gates = detect_gate_corners(bgr)
        elapsed_ms = (time.perf_counter() - t0) / N * 1000

        print(f'{os.path.basename(img_path)}: '
              f'{len(gates)} gate(s) detected  |  '
              f'{elapsed_ms:.2f} ms/frame  '
              f'({1000/elapsed_ms:.0f} fps equivalent)')

        for i, corners in enumerate(gates):
            print(f'  Gate {i}: corners = {corners.tolist()}')

        # Show annotated image
        out = draw_corners(bgr, gates)
        cv2.imshow(os.path.basename(img_path), out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
