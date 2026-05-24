import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture(2)

# allow manual resizing and set an initial large size
cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Tuner", 800, 400)
# mask window too
cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mask", 800, 400)

# ---- HSV sliders (Range 1) ----
cv2.createTrackbar("H min 1", "HSV Tuner", 0, 180, nothing)
cv2.createTrackbar("H max 1", "HSV Tuner", 180, 180, nothing)
cv2.createTrackbar("S min 1", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("S max 1", "HSV Tuner", 60, 255, nothing)
cv2.createTrackbar("V min 1", "HSV Tuner", 180, 255, nothing)
cv2.createTrackbar("V max 1", "HSV Tuner", 255, 255, nothing)

# ---- HSV sliders (Range 2 - for wraparound colors like red) ----
cv2.createTrackbar("H min 2", "HSV Tuner", 170, 180, nothing)
cv2.createTrackbar("H max 2", "HSV Tuner", 180, 180, nothing)
cv2.createTrackbar("S min 2", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("S max 2", "HSV Tuner", 255, 255, nothing)
cv2.createTrackbar("V min 2", "HSV Tuner", 0, 255, nothing)
cv2.createTrackbar("V max 2", "HSV Tuner", 255, 255, nothing)

print("Adjust sliders until only the ball is visible in mask.")
print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    # Brightness / contrast adjustment for visualization
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=25)
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    display = frame.copy()

    # ---- read sliders (Range 1) ----
    hmin1 = cv2.getTrackbarPos("H min 1", "HSV Tuner")
    hmax1 = cv2.getTrackbarPos("H max 1", "HSV Tuner")
    smin1 = cv2.getTrackbarPos("S min 1", "HSV Tuner")
    smax1 = cv2.getTrackbarPos("S max 1", "HSV Tuner")
    vmin1 = cv2.getTrackbarPos("V min 1", "HSV Tuner")
    vmax1 = cv2.getTrackbarPos("V max 1", "HSV Tuner")

    # ---- read sliders (Range 2) ----
    hmin2 = cv2.getTrackbarPos("H min 2", "HSV Tuner")
    hmax2 = cv2.getTrackbarPos("H max 2", "HSV Tuner")
    smin2 = cv2.getTrackbarPos("S min 2", "HSV Tuner")
    smax2 = cv2.getTrackbarPos("S max 2", "HSV Tuner")
    vmin2 = cv2.getTrackbarPos("V min 2", "HSV Tuner")
    vmax2 = cv2.getTrackbarPos("V max 2", "HSV Tuner")

    # overlay values on display
    text1 = f"Range 1: H:[{hmin1},{hmax1}] S:[{smin1},{smax1}] V:[{vmin1},{vmax1}]"
    text2 = f"Range 2: H:[{hmin2},{hmax2}] S:[{smin2},{smax2}] V:[{vmin2},{vmax2}]"
    cv2.putText(display, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # create masks for both ranges and combine
    lower1 = np.array([hmin1, smin1, vmin1])
    upper1 = np.array([hmax1, smax1, vmax1])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    lower2 = np.array([hmin2, smin2, vmin2])
    upper2 = np.array([hmax2, smax2, vmax2])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # combine both masks (useful for colors that wrap around, like red)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # ---- contour detection ----
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    display = frame.copy()

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(display, (cx, cy), 8, (0, 255, 0), -1)
            cv2.circle(display, (int(x), int(y)), int(radius),
                       (255, 0, 0), 2)

    cv2.imshow("HSV Tuner", display)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal HSV:")
print(f"Lower = ({hmin}, {smin}, {vmin})")
print(f"Upper = ({hmax}, {smax}, {vmax})")