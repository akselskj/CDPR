import cv2
import numpy as np
import time
import parameters as p
from collections import deque
import threading
import sys, io


class CameraCapture:
    def __init__(self, device=2):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self._last_good_frame = None
        self._stop = threading.Event()
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

    def _grab_loop(self):
        while not self._stop.is_set():
            self.cap.grab()
            time.sleep(0)

    def read(self):
        # Intercept stderr to catch corrupt JPEG warnings
        buf = io.StringIO()
        real_stderr, sys.stderr = sys.stderr, buf

        ret, frame = self.cap.retrieve()

        sys.stderr = real_stderr
        corrupt = "Corrupt JPEG" in buf.getvalue()

        if ret and frame is not None and frame.size > 0 and not corrupt:
            self._last_good_frame = frame
            return True, frame
        return False, None

    def release(self):
        self._stop.set()
        self._grab_thread.join()
        self.cap.release()





class BallTracker:
    def __init__(self, H, t0, buffer=64, y_target=-0.05):
        """
        H : pixel -> plane homography (plane units = meters)
        t0: global start time (perf_counter)
        y_target: interception line in plane coords (meters)
        """
        self.H = H
        self.t0 = t0

        # track green ball:
        #self.colorLower = (29, 86, 6)
        #self.colorUpper = (64, 255, 255)

        # track white ball:
        #self.colorLower = (7, 0, 160)
        #self.colorUpper = (60, 60, 255)

        # red
        self.redLower1 = (0, 50, 50)
        self.redUpper1 = (15, 255, 255)

        self.redLower2 = (185, 50, 50)
        self.redUpper2 = (255, 255, 255)

        self.pts = deque(maxlen=buffer)
        self.y_target = y_target

        # plane y_target line → pixel row
        plane_pt = np.array([[[0.0, self.y_target]]], dtype=np.float32)
        px_pt = cv2.perspectiveTransform(plane_pt, np.linalg.inv(self.H))[0][0]

        self.y_target_px = float(px_pt[1])

        # ---- TRACKED STATE ----

        self.center = None

        # filtered plane position/velocity
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)

        # raw measurements
        self.pos_raw = np.zeros(2)
        self.vel_raw = np.zeros(2)

        # previous state
        self.prev_pos = np.zeros(2)
        self.vel_prev = np.zeros(2)

        # filtering
        self.alpha_pos = 0.3
        self.alpha_vel = 0.1

        # tracking quality
        self.frames_without_ball = 0

        # prediction
        self.p_hit = None
        self.t_hit = None

        # visualization
        self.hit_px = None
        self.traj_px = None

    # ---- pixel → plane (meters) ----
    def pixel_to_plane(self, px):
        pt = np.array([[[px[0], px[1]]]], dtype=np.float32)
        plane = cv2.perspectiveTransform(pt, self.H)[0][0]
        return plane

    # ---- detect ball center ----
    def detect_ball(self, frame):
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.redLower1, self.redUpper1)
        mask2 = cv2.inRange(hsv, self.redLower2, self.redUpper2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if len(cnts) == 0:
            return None
        
        best_center = None
        best_radius = 0

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 30:
                continue

            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue

            #circularity = 4 * np.pi * area / (peri * peri)
            #if circularity < 0.7:
            #    continue

            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius < 5:
                continue

            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # ---- plane gate ----
            plane = self.pixel_to_plane((cx, cy))
            if not ((-0.5 <= plane[0] <= 0.5) and (-0.5 <= plane[1] <= 1)):
                continue

            if radius > best_radius:
                best_radius = radius
                best_center = (cx, cy)

        return best_center


    # ---- update + predict hit ----
    def update(self, frame, dt):

        t = time.perf_counter() - self.t0

        # ---- DETECTION ----

        center = self.detect_ball(frame)
        self.center = center

        if center is None:

            self.frames_without_ball += 1
            self.traj_px = None
            self.hit_px = None
            self.p_hit = None
            self.t_hit = None

            return

        self.frames_without_ball = 0

        # ---- STORE HISTORY ----

        self.pts.appendleft((t - p.cam_latency, center))

        # ---- POSITION ESTIMATION ----

        pos_raw = self.pixel_to_plane(center)
        self.pos_raw = pos_raw

        # latency compensation
        pos_pred = pos_raw + self.vel_prev * p.cam_latency

        # filtering
        self.pos = (
            self.alpha_pos * pos_pred
            + (1 - self.alpha_pos) * self.prev_pos
        )

        # ---- VELOCITY ESTIMATION ----

        dt = max(dt, 1e-3)
        vel_raw = (self.pos - self.prev_pos) / dt
        self.vel_raw = vel_raw.copy()

        if np.linalg.norm(vel_raw) > p.MAX_VEL:
            vel_raw = self.vel_prev

        vel = (
            self.alpha_vel * vel_raw
            + (1 - self.alpha_vel) * self.vel_prev
        )

        self.vel = vel.copy()

        # ---- STORE STATE ----

        self.prev_pos = self.pos.copy()
        self.vel_prev = self.vel.copy()

        # ---- HIT PREDICTION ----

        self._predict_hit(frame, t)
    

    def _predict_hit(self, frame, t):
        """
        Predict future ball impact point using polynomial fit in pixel space.

        Updates:
            self.p_hit
            self.t_hit
            self.hit_px
            self.traj_px
        """

        # ---- REQUIRE ENOUGH HISTORY ----

        valid = [p for p in self.pts if p is not None]

        if len(valid) < 20:
            self.p_hit = None
            self.t_hit = None
            self.hit_px = None
            self.traj_px = None
            return

        sample = valid[:20]

        # ---- EXTRACT HISTORY ----

        times = np.array([p[0] for p in sample])
        xs = np.array([p[1][0] for p in sample])
        ys = np.array([p[1][1] for p in sample])

        # relative time improves numerical conditioning
        times = times - times[0]

        # ---- POLYNOMIAL FIT ----

        try:
            x_coef = np.polyfit(times, xs, 1)
            y_coef = np.polyfit(times, ys, 2)

        except np.linalg.LinAlgError:

            self.p_hit = None
            self.t_hit = None
            self.hit_px = None
            self.traj_px = None

            return

        # ---- INTERSECTION WITH TARGET LINE ----

        b2, b1, b0 = y_coef

        roots = np.roots([
            b2,
            b1,
            b0 - self.y_target_px
        ])

        # future physical roots only
        future = [
            r.real
            for r in roots
            if np.isreal(r) and 0 < r.real < 1.5
        ]

        if len(future) == 0:

            self.p_hit = None
            self.t_hit = None
            self.hit_px = None
            self.traj_px = None

            return

        t_future = min(future)

        # ---- PREDICT IMPACT X ----

        a1, a0 = x_coef

        x_future = a1 * t_future + a0

        # ---- GENERATE TRAJECTORY OVERLAY ----

        traj_px = []

        t_samples = np.linspace(0, t_future, 30)

        for ts in t_samples:

            x_pred = a1 * ts + a0
            y_pred = b2 * ts**2 + b1 * ts + b0

            traj_px.append((
                int(x_pred),
                int(y_pred)
            ))

        # ---- IMAGE BOUNDS CHECK ----

        h, w = frame.shape[:2]

        if not (0 <= x_future < w):

            self.p_hit = None
            self.t_hit = None
            self.hit_px = None
            self.traj_px = None

            return

        # ---- PIXEL -> PLANE ----

        plane = self.pixel_to_plane((
            x_future,
            self.y_target_px
        ))

        p_hit = np.array([
            plane[0],
            plane[1]
        ])

        # ---- WORKSPACE SANITY CHECK ----

        if not (
            p.WORKSPACE_X_MIN < p_hit[0] < p.WORKSPACE_X_MAX
            and
            p.WORKSPACE_Y_MIN < p_hit[1] < p.WORKSPACE_Y_MAX
        ):

            self.p_hit = None
            self.t_hit = None
            self.hit_px = None
            self.traj_px = None

            return

        # ---- STORE PREDICTION ----

        t_hit = times[0] + t_future + t

        hit_px = (
            x_future,
            self.y_target_px
        )

        self.p_hit = p_hit
        self.t_hit = t_hit

        self.hit_px = hit_px
        self.traj_px = traj_px
    

    
    def draw_overlay(self, frame, desired_pos_px = None):
        if self.center is not None:
            cv2.circle(frame, self.center, 8, (0, 255, 0), -1)

        # ---- measured trajectory history ----
        valid = [p for p in self.pts if p is not None]

        for i in range(len(valid) - 1):

            p1 = valid[i][1]
            p2 = valid[i + 1][1]

            cv2.line(frame, p1, p2, (0, 100, 255), 2)

        if self.hit_px is not None:
            cv2.circle(frame, (int(self.hit_px[0]), int(self.hit_px[1])), 8, (255, 0, 0), -1)

        # ---- predicted trajectory ----
        if self.traj_px is not None and len(self.traj_px) > 1:

            for i in range(len(self.traj_px) - 1):
                cv2.line(
                    frame,
                    self.traj_px[i],
                    self.traj_px[i + 1],
                    (255, 255, 0),
                    2
                )

        # Draw desired position from controller (magenta cross)
        if desired_pos_px is not None:
            x, y = int(desired_pos_px[0]), int(desired_pos_px[1])
            cv2.drawMarker(frame, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 15, 2)

        # origin
        origin_px = cv2.perspectiveTransform(
            np.array([[[0.0, 0.0]]], dtype=np.float32),
            np.linalg.inv(self.H)
        )[0][0]
        cv2.circle(frame, tuple(origin_px.astype(int)), 6, (0, 0, 255), -1)

        # ---- draw workspace bounds (-0.5..0.5 m) ----
        xs = [-0.5, 0.5]
        ys = [-0.5, 1]

        corners_plane = np.array([
            [xs[0], ys[0]],
            [xs[1], ys[0]],
            [xs[1], ys[1]],
            [xs[0], ys[1]],
        ], dtype=np.float32)

        corners_px = cv2.perspectiveTransform(
            corners_plane[None, :, :],
            np.linalg.inv(self.H)
        )[0]

        for i in range(4):
            p1 = tuple(corners_px[i].astype(int))
            p2 = tuple(corners_px[(i+1)%4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)






def calibrate_plane(cap, duration=2.0):
    """
    Interactive ArUco plane calibration.
    Returns pixel->plane homography H (meters).
    """

    MARKER_SIZE = 0.074  # meters

    anchors = {
        0: ("BL", np.array([-0.650, -0.460])),
        1: ("TL", np.array([-0.644,  0.220])),
        2: ("TR", np.array([ 0.645,  0.221])),
        3: ("BR", np.array([ 0.649, -0.459])),
    }

    # ---- build plane corners ----
    plane_points = {}
    for mid, (corner_type, anchor) in anchors.items():

        if corner_type == "TL":
            TL = anchor
            TR = anchor + [ MARKER_SIZE, 0]
            BR = anchor + [ MARKER_SIZE,-MARKER_SIZE]
            BL = anchor + [ 0,-MARKER_SIZE]

        elif corner_type == "TR":
            TR = anchor
            TL = anchor + [-MARKER_SIZE, 0]
            BL = anchor + [-MARKER_SIZE,-MARKER_SIZE]
            BR = anchor + [ 0,-MARKER_SIZE]

        elif corner_type == "BL":
            BL = anchor
            BR = anchor + [ MARKER_SIZE, 0]
            TR = anchor + [ MARKER_SIZE, MARKER_SIZE]
            TL = anchor + [ 0, MARKER_SIZE]

        elif corner_type == "BR":
            BR = anchor
            BL = anchor + [-MARKER_SIZE, 0]
            TL = anchor + [-MARKER_SIZE, MARKER_SIZE]
            TR = anchor + [ 0, MARKER_SIZE]

        plane_points[mid] = np.array([TL, TR, BR, BL], dtype=np.float32)

    # ---- ArUco detector ----
    aruco = cv2.aruco
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
    detector = aruco.ArucoDetector(dict_aruco)

    print("Show all 4 markers. Press 'y' when ready.")

    # ---- wait for user confirmation ----
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # temporary frame estimate from visible markers
        if ids is not None and len(ids) >= 2:

            img_pts = []
            world_pts = []

            for i, mid in enumerate(ids.flatten()):
                if mid in plane_points:
                    img_pts.extend(corners[i][0])
                    world_pts.extend(plane_points[mid])

            if len(img_pts) >= 8:

                H_tmp, _ = cv2.findHomography(
                    np.array(img_pts, dtype=np.float32),
                    np.array(world_pts, dtype=np.float32),
                    0
                )

                if H_tmp is not None:
                    draw_plane_frame(frame, np.linalg.inv(H_tmp))

        cv2.imshow("Plane calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("y"):
            break

    print("Collecting calibration frames...")

    # ---- collect correspondences ----
    img_pts_all = []
    world_pts_all = []

    t_end = time.perf_counter() + duration

    while time.perf_counter() < t_end:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        for i, mid in enumerate(ids.flatten()):
            if mid in plane_points:
                img_pts_all.extend(corners[i][0])
                world_pts_all.extend(plane_points[mid])

        # optional display
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Plane calibration", frame)
        cv2.waitKey(1)

    img_pts_all = np.array(img_pts_all, dtype=np.float32)
    world_pts_all = np.array(world_pts_all, dtype=np.float32)

    if len(img_pts_all) < 14:
        raise RuntimeError("Not enough marker detections for calibration")

    # ---- robust homography ----
    H, mask = cv2.findHomography(img_pts_all, world_pts_all, cv2.RANSAC)

    print("Plane calibration complete")

    return H


def draw_plane_frame(frame, H_inv,
                     x_range=(-0.7, 0.7),
                     y_range=(-0.5, 0.5),
                     grid_spacing=0.1,
                     axis_len=0.15):

    """
    Draw projected plane coordinate frame + grid.

    x_range, y_range : grid limits in plane coordinates [m]
    grid_spacing     : spacing between grid lines [m]
    """

    # ---- GRID ----

    xs = np.arange(x_range[0], x_range[1] + 1e-6, grid_spacing)
    ys = np.arange(y_range[0], y_range[1] + 1e-6, grid_spacing)

    # vertical grid lines
    for x in xs:

        pts_plane = np.array([[
            [x, y_range[0]],
            [x, y_range[1]]
        ]], dtype=np.float32)

        pts_img = cv2.perspectiveTransform(pts_plane, H_inv)[0]

        p1 = tuple(pts_img[0].astype(int))
        p2 = tuple(pts_img[1].astype(int))

        cv2.line(frame, p1, p2, (80, 80, 80), 1)

    # horizontal grid lines
    for y in ys:

        pts_plane = np.array([[
            [x_range[0], y],
            [x_range[1], y]
        ]], dtype=np.float32)

        pts_img = cv2.perspectiveTransform(pts_plane, H_inv)[0]

        p1 = tuple(pts_img[0].astype(int))
        p2 = tuple(pts_img[1].astype(int))

        cv2.line(frame, p1, p2, (80, 80, 80), 1)

    # ---- AXES ----

    pts_plane = np.array([[
        [0.0, 0.0],
        [axis_len, 0.0],
        [0.0, axis_len]
    ]], dtype=np.float32)

    pts_img = cv2.perspectiveTransform(pts_plane, H_inv)[0]

    o  = tuple(pts_img[0].astype(int))
    px = tuple(pts_img[1].astype(int))
    py = tuple(pts_img[2].astype(int))

    # x-axis
    cv2.arrowedLine(
        frame, o, px,
        (0, 0, 255),
        2,
        tipLength=0.08
    )

    # y-axis
    cv2.arrowedLine(
        frame, o, py,
        (0, 255, 0),
        2,
        tipLength=0.08
    )

    # origin
    cv2.circle(frame, o, 4, (255,255,255), -1)

    # labels
    cv2.putText(
        frame, "x",
        (px[0] + 5, px[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,255),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame, "y",
        (py[0] + 5, py[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,255,0),
        1,
        cv2.LINE_AA
    )


def find_platform(cap, H, duration=2.0, EE_ID=1):

    aruco = cv2.aruco
    dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
    detector = aruco.ArucoDetector(dict_aruco)

    print("Show EE marker. Press 'y' when ready.")

    # ---- wait for user confirmation ----
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Platform calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y"):
            break

    print("Collecting platform pose samples...")

    samples = []

    t_end = time.perf_counter() + duration

    while time.perf_counter() < t_end:

        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        for i, mid in enumerate(ids.flatten()):

            if mid == EE_ID:

                pts = corners[i][0]

                # ---- pixel center ----
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])

                # ---- pixel -> plane ----
                pixel_pt = np.array([cx, cy, 1.0])
                world_pt = H @ pixel_pt
                world_pt /= world_pt[2]

                x = world_pt[0]
                y = world_pt[1]

                # ---- orientation from marker ----
                v = pts[1] - pts[0]  # top-left to top-right
                theta = np.arctan2(-v[1], v[0])  # minus because image y downward

                samples.append([x, y, theta])

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Platform calibration", frame)
        cv2.waitKey(1)

    cv2.destroyWindow("Platform calibration")

    if len(samples) < 5:
        raise RuntimeError("Not enough EE detections for platform calibration")

    samples = np.array(samples)

    # ---- average pose ----
    x_home = np.mean(samples[:, 0])
    y_home = np.mean(samples[:, 1])

    # Proper circular mean for theta
    sin_mean = np.mean(np.sin(samples[:, 2]))
    cos_mean = np.mean(np.cos(samples[:, 2]))
    theta_home = np.arctan2(sin_mean, cos_mean)

    print("Platform home pose:")
    print("x:", x_home, "y:", y_home, "theta:", theta_home)

    return np.array([x_home, y_home, theta_home])
