"""Offline demonstration of queue metrics without a video or camera.

Uses synthetic track IDs and a simulated clock to exercise the dwell-time logic.
This demonstrates the logic only; it does not measure real-world waiting time.
Run the PowerShell command shown below.
"""

import json

from checkout_queue_monitor import QueueDwellMonitor


# A simple square ROI and foot-point coordinates inside it.
monitor = QueueDwellMonitor(
    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    alert_after_sec=60.0,
)
checkpoints = {0, 10, 30, 59, 60, 70}

for second in range(71):
    people = {}
    if second < 70:
        people[1001] = (50, 80)  # Person 1001 waits for 70 simulated seconds.
    if 10 <= second < 40:
        people[1002] = (65, 80)  # Person 1002 joins at 10s and leaves at 40s.

    metrics = monitor.update(people, now=float(second))
    if second in checkpoints:
        print(json.dumps({"sim_time_sec": second, **metrics}))

