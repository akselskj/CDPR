# Vision Pipeline Visualization Script

import cv2
import numpy as np

# =========================
# Camera setup
# =========================
cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# =========================
# HSV range for red ball
# Adjust these values!
# =========================
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

kernel = np.ones((5, 5), np.uint8)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Keep original frame for processing
    process_frame = frame.copy()

    # Brightened frame only for display
    display_frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=25)

    if not ret:
        break

    # =========================
    # Step 1: Original frame
    # =========================
    original = display_frame.copy()

    # =========================
    # Step 2: Gaussian blur
    # =========================
    blurred_display = cv2.GaussianBlur(display_frame, (11, 11), 0)
    blurred = cv2.GaussianBlur(process_frame, (11, 11), 0)

    # =========================
    # Step 3: HSV conversion
    # =========================
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # =========================
    # Step 4: Thresholding
    # =========================
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # =========================
    # Step 5: Morphological filtering
    # =========================
    mask_clean = cv2.erode(mask, kernel, iterations=2)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)

    # =========================
    # Step 6: Contour detection
    # =========================
    contour_img = display_frame.copy()

    contours, _ = cv2.findContours(
        mask_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        c = max(contours, key=cv2.contourArea)

        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)
            cv2.circle(contour_img, (int(x), int(y)), int(radius), (255, 0, 0), 2)
            cv2.circle(contour_img, (int(x), int(y)), 5, (0, 0, 255), -1)

    # =========================
    # Convert grayscale masks to BGR
    # =========================
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_clean_bgr = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)

    # =========================
    # Labels
    # =========================
    def label(img, text):
        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return img

    original = label(original, "Original")
    blurred = label(blurred, "Gaussian Blur")
    # =========================
    # HSV channel visualization
    # =========================

    h, s, v = cv2.split(hsv)

    # Normalize for display
    h_vis = cv2.normalize(h, None, 0, 255, cv2.NORM_MINMAX)
    s_vis = cv2.normalize(s, None, 0, 255, cv2.NORM_MINMAX)
    v_vis = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to BGR
    h_vis = cv2.cvtColor(h_vis, cv2.COLOR_GRAY2BGR)
    s_vis = cv2.cvtColor(s_vis, cv2.COLOR_GRAY2BGR)
    v_vis = cv2.cvtColor(v_vis, cv2.COLOR_GRAY2BGR)

    # =========================
    # Larger labels
    # =========================

    def big_label(img, text):

        # black background rectangle
        cv2.rectangle(img, (10, 10), (320, 80), (0, 0, 0), -1)

        cv2.putText(
            img,
            text,
            (25, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,          # font scale
            (255, 255, 255),
            5,            # thickness
            cv2.LINE_AA,
        )

        return img

    h_vis = big_label(h_vis, "Hue")
    s_vis = big_label(s_vis, "Saturation")
    v_vis = big_label(v_vis, "Value")

    # =========================
    # Resize channels
    # =========================

    height, width = frame.shape[:2]

    small_w = width // 2
    small_h = height // 2

    h_vis = cv2.resize(h_vis, (small_w, small_h))
    s_vis = cv2.resize(s_vis, (small_w, small_h))
    v_vis = cv2.resize(v_vis, (small_w, small_h))

    # Empty black image
    blank = np.zeros_like(v_vis)

    # Top row
    top_row = np.hstack([h_vis, s_vis])

    # Bottom row with centered Value image
    side_blank = np.zeros((small_h, small_w // 2, 3), dtype=np.uint8)

    bottom_row = np.hstack([
        side_blank,
        v_vis,
        side_blank
    ])

    # Crop back to correct width
    bottom_row = bottom_row[:, :width]

    # Final HSV panel
    hsv_vis = np.vstack([top_row, bottom_row])

    mask_bgr = label(mask_bgr, "Threshold")
    mask_clean_bgr = label(mask_clean_bgr, "Morphology")
    contour_img = label(contour_img, "Contour Detection")

    # =========================
    # Resize for display
    # =========================
    scale = 0.4

    def resize(img):
        return cv2.resize(img, None, fx=scale, fy=scale)

    top = np.hstack([
        resize(original),
        resize(blurred_display),
        resize(hsv_vis)
    ])

    bottom = np.hstack([
        resize(mask_bgr),
        resize(mask_clean_bgr),
        resize(contour_img)
    ])

    combined = np.vstack([top, bottom])

    cv2.imshow("Vision Pipeline", combined)

    key = cv2.waitKey(1)

    # Press q to quit
    if key == ord('q'):
        break

    # Press s to save figure
    if key == ord('s'):
        cv2.imwrite("vision_pipeline.png", combined)
        print("Saved vision_pipeline.png")

cap.release()
cv2.destroyAllWindows()

