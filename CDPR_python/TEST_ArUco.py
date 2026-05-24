import cv2
import numpy as np
import time
import camera as cam

cap = cam.CameraCapture(device=2)
time.sleep(0.5)

for p in range(10):
    start = time.perf_counter()
    for i in range(100):
        ret, frame = cap.read()
    end = time.perf_counter()
    print("True fps: ", 100/(end-start))
cap.release()



cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

H = cam.calibrate_plane(cap, duration=2.0)
cap.release()
time.sleep(0.5)

t0 = time.perf_counter()

tracker = cam.BallTracker(H, t0)

while True:
    t0 = time.perf_counter()
    ret, frame = cap.read()
    t1 = time.perf_counter()
    center = tracker.detect_ball(frame)
    t2 = time.perf_counter()
    tracker.draw_overlay(frame, center, None, None)
    t3 = time.perf_counter()
    print(f"read={1000*(t1-t0):.1f}ms  detect={1000*(t2-t1):.1f}ms  draw={1000*(t3-t2):.1f}ms")