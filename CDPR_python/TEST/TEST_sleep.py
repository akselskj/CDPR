import time
import numpy as np
import matplotlib.pyplot as plt

DURATION = 5.0
TARGET_DT = 0.02


def run_sleep():
    t_log, dt_log = [], []
    t0 = time.perf_counter()
    last = t0

    while True:
        now = time.perf_counter()
        if now - t0 > DURATION:
            break

        dt = now - last
        last = now

        t_log.append(now - t0)
        dt_log.append(dt)

        time.sleep(TARGET_DT)

    return np.array(t_log), np.array(dt_log)


def run_hybrid():
    t_log, dt_log = [], []
    t0 = time.perf_counter()
    last = t0
    next_t = t0 + TARGET_DT

    while True:
        now = time.perf_counter()
        if now - t0 > DURATION:
            break

        dt = now - last
        last = now

        t_log.append(now - t0)
        dt_log.append(dt)

        while True:
            now = time.perf_counter()
            if now >= next_t:
                break
            if next_t - now > 0.001:
                time.sleep(0)

        next_t += TARGET_DT

    return np.array(t_log), np.array(dt_log)


def run_no_sleep():
    t_log, dt_log = [], []
    t0 = time.perf_counter()
    last = t0

    while True:
        now = time.perf_counter()
        if now - t0 > DURATION:
            break

        dt = now - last
        last = now

        t_log.append(now - t0)
        dt_log.append(dt)

    return np.array(t_log), np.array(dt_log)


# Run tests
t_sleep, dt_sleep = run_sleep()
print("sleep run finished")
t_hybrid, dt_hybrid = run_hybrid()
print("hybrid run finished")
t_nosleep, dt_nosleep = run_no_sleep()

# Convert to ms
dt_sleep *= 1000
dt_hybrid *= 1000
dt_nosleep *= 1000


# Plot
plt.figure(figsize=(12, 6))

plt.plot(t_sleep, dt_sleep, label="sleep(target_dt)", alpha=0.8)
plt.plot(t_hybrid, dt_hybrid, label="hybrid yield/spin", alpha=0.8)

# downsample no-sleep for readability
stride = max(1, len(t_nosleep) // 2000)
plt.plot(t_nosleep[::stride], dt_nosleep[::stride],
         label="no sleep (downsampled)", alpha=0.6)

plt.axhline(TARGET_DT * 1000, color="k", linestyle="--", label="target dt")

plt.xlabel("Time [s]")
plt.ylabel("dt [ms]")
plt.title("Loop dt vs time (5 seconds)")
plt.legend()
plt.grid(True)

plt.show()


# Print stats
def stats(name, data):
    print(f"{name:20s} "
          f"mean={data.mean():6.2f} ms  "
          f"min={data.min():6.2f} ms  "
          f"max={data.max():6.2f} ms")

print("\nTiming statistics:")
stats("sleep", dt_sleep)
stats("hybrid", dt_hybrid)
stats("no sleep", dt_nosleep)

