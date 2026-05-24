import cv2
import time
import queue
import threading
import numpy as np
import camera as cam


def camera_tracking_loop(cap, tracker, shared_hit, frame_queue=None):
    # thread will open its own capture (caller should pass device index)
    cap = cv2.VideoCapture(2)

    print(f"Camera thread started (dev={cap}) - cap.isOpened()={cap.isOpened()}")

    read_fail_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            read_fail_count += 1
            if read_fail_count % 50 == 0:
                print(f"DEBUG: camera read failing (count={read_fail_count}) cap.isOpened()={cap.isOpened()}")
            time.sleep(0.01)
            continue
        read_fail_count = 0

        p_hit, t_hit, ball_px, hit_px = tracker.update_bounce(frame)

        if p_hit is not None:
            shared_hit.set(p_hit, t_hit)
            print(p_hit)

        # ---- draw ball ----
        if ball_px is not None:
            cv2.circle(frame, ball_px, 8, (0, 255, 0), -1)

        # ---- draw predicted hit ----
        if hit_px is not None:
            cv2.circle(frame, (int(hit_px[0]), int(hit_px[1])), 8, (255, 0, 0), -1)

        # ---- draw origin ----
        origin_px = cv2.perspectiveTransform(
            np.array([[[0.0, 0.0]]], dtype=np.float32),
            np.linalg.inv(tracker.H)
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
            np.linalg.inv(tracker.H)
        )[0]

        for i in range(4):
            p1 = tuple(corners_px[i].astype(int))
            p2 = tuple(corners_px[(i+1)%4].astype(int))
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

        # send annotated frame to main thread for display
        if frame_queue is not None:
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass


class SharedHit:
    def __init__(self):
        self.lock = threading.Lock()
        self.p_hit = None
        self.t_hit = None

    def set(self, p_hit, t_hit):
        with self.lock:
            self.p_hit = p_hit.copy()
            self.t_hit = float(t_hit)

    def get(self):
        with self.lock:
            return self.p_hit, self.t_hit


def ping_pong_bot():
    cap = cv2.VideoCapture(2)

    t0 = time.perf_counter()

    # ---- plane calibration ----
    H = cam.calibrate_plane(cap, duration=2.0)

    # diagnostic: ensure capture is still open after calibration
    print("DEBUG: cap.isOpened() after calibration:", cap.isOpened())
    ret_check, _ = cap.read()
    print("DEBUG: sample read after calibration returned:", ret_check)

    # release the main capture so the camera thread can open the device
    cap.release()
    cv2.destroyWindow("Plane calibration")
    time.sleep(0.5)

    shared_hit = SharedHit()

    tracker = cam.BallTracker(H, t0)

    # ---- start camera thread ----
    frame_queue = queue.Queue(maxsize=1)

    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(2, tracker, shared_hit, frame_queue),
        daemon=True,
    )
    cam_thread.start()

    # ---- main display loop (owns GUI) ----
    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow("Robot vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass

ping_pong_bot()