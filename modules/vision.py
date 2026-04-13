import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HSV thresholds – pink/magenta wraps around hue 0 / 179
# S >= 60 keeps saturated panel colour; V >= 40 works at distance.
# ---------------------------------------------------------------------------
HSV_LO1 = np.array([140,  60,  40])
HSV_HI1 = np.array([179, 255, 255])
HSV_LO2 = np.array([  0,  60,  40])
HSV_HI2 = np.array([ 10, 255, 255])

BORDER       = 8    # px – strip camera frame (pure-magenta border) before detection
MIN_AREA     = 200  # px² – reject noise / small distant gates are still ≥ this
MIN_SOLIDITY = 0.55 # area / convex-hull-area – panels are convex; relax for transparency
MAX_ASPECT   = 2.5  # max(w,h)/min(w,h) – square panel viewed at an angle can reach ~2.4
MIN_PIXEL_H  = 6    # px – minimum apparent height; below this depth estimate is unreliable


def detect_gates(image):
    """
    Detect pink/magenta gate panels in a BGR image.

    Works in both the Webots simulation (semi-transparent emissive panels) and
    real-world images where the panels are solid magenta.

    Parameters
    ----------
    image : np.ndarray
        BGR image (e.g. from cv2.imread or converted from BGRA camera feed).

    Returns
    -------
    list of dict, one entry per detected gate panel:
        'corners'  : (4, 2) float32 – corners of the min-area bounding rectangle
        'centroid' : (cx, cy) float  – pixel centroid of the panel
        'pixel_h'  : float           – estimated vertical pixel span of the panel
        'bbox'     : (x, y, w, h)   – axis-aligned bounding box
        'area'     : float           – contour area in pixels²
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, HSV_LO1, HSV_HI1),
        cv2.inRange(hsv, HSV_LO2, HSV_HI2),
    )

    # Strip camera frame – the Webots camera adds a pure-magenta 1-pixel border
    # that bleeds inward a few pixels; masking it avoids false contours at edges.
    mask[:BORDER,  :]  = 0
    mask[-BORDER:, :]  = 0
    mask[:,  :BORDER]  = 0
    mask[:, -BORDER:]  = 0

    # 7×7 morphological close: fills the dark interior of semi-transparent panels
    # without merging separate gates (they are always >7 px apart in practice).
    k    = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        # Reject non-convex blobs (tree leaves, floor patterns that sneak through).
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area < 1 or (area / hull_area) < MIN_SOLIDITY:
            continue

        # Minimum-area rectangle: always gives exactly 4 corners (unlike
        # approxPolyDP which can return 3, 5, … on partially occluded panels).
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if min(rw, rh) < 1:
            continue
        if max(rw, rh) / (min(rw, rh) + 1e-6) > MAX_ASPECT:
            continue

        corners = cv2.boxPoints(rect).astype(np.float32)  # (4, 2)

        # Centroid = mean of the four rectangle corners.
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))

        # Pixel height: mean of the two bottom-most Y values minus mean of the
        # two top-most Y values (robust to slight rotation of the rectangle).
        sorted_y = corners[np.argsort(corners[:, 1])]
        pixel_h  = float(np.mean(sorted_y[2:, 1]) - np.mean(sorted_y[:2, 1]))
        if pixel_h < MIN_PIXEL_H:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        results.append({
            'corners':  corners,
            'centroid': (cx, cy),
            'pixel_h':  pixel_h,
            'bbox':     (x, y, bw, bh),
            'area':     area,
        })

    # Largest panel first (most useful for navigation when multiple are visible).
    results.sort(key=lambda d: d['area'], reverse=True)
    return results


def draw_detections(image, detections):
    """
    Return a copy of *image* with detection overlays drawn on it.

    Draws a green quadrilateral around each detected panel, a red dot at the
    centroid, and a label showing pixel height and area.
    """
    out = image.copy()
    for i, det in enumerate(detections):
        pts  = det['corners'].astype(np.int32).reshape(-1, 1, 2)
        cx   = int(det['centroid'][0])
        cy   = int(det['centroid'][1])

        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        label = f"#{i+1}  h={det['pixel_h']:.0f}px  A={det['area']:.0f}"
        cv2.putText(out, label, (cx - 30, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Quick test: run  python vision.py  from the modules/ directory
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os, sys

    pics_dir = os.path.join(os.path.dirname(__file__), 'vision_pics')
    if not os.path.isdir(pics_dir):
        print(f'vision_pics/ not found at {pics_dir}')
        sys.exit(1)

    for fname in sorted(os.listdir(pics_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(pics_dir, fname)
        img  = cv2.imread(path)
        if img is None:
            print(f'Could not read {path}')
            continue

        dets = detect_gates(img)
        out  = draw_detections(img, dets)

        print(f'{fname}: {len(dets)} gate(s) detected')
        for i, d in enumerate(dets):
            cx, cy = d['centroid']
            print(f'  [{i+1}] centroid=({cx:.1f},{cy:.1f})  '
                  f'pixel_h={d["pixel_h"]:.1f}  area={d["area"]:.0f}')

        cv2.imshow(fname, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
