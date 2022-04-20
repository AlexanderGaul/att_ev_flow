import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.preprocessing.event_functional import event_array_patch_context, event_array_patch_time_surface

res = (5, 5)
dt = 1.
events_raw = np.array([[0, 0, 0, 1],
                   [0, 1, 0.1, 1],
                   [1, 1, 0.1, 1],
                   [1, 1, 0.1, -1],
                   [2, 2, 0.2, -1],
                   [2, 2, 0.2, -1],
                   [2, 3, 0.3, 1],
                   [3, 3, 0.3, -1],
                   [4, 4, 0.4, -1],
                   [0, 0, 0.5, 1],
                   [0, 2, 0.6, 1],
                   [3, 1, 0.7, 1]])

events_context = event_array_patch_context(events_raw, res, dt, 2, 3)
print(events_context)

events = events_raw.copy()
events[:, 2] = events_raw[:, 2] / dt * 2 - 1

assert (events_context[0, ] == 0.).all()
assert (events_context[1, 1 +18] == events[0, 2])
assert (events_context[1, 1] == events[0, 3])

assert (events_context[2, 0 +18] == events[0, 2])
assert (events_context[2, 3 +18] == events[1, 2])

assert (events_context[2, 0] == events[0, 3])
assert (events_context[2, 3] == events[1, 3])

assert (events_context[3, 0 +18] == events[0, 2])
assert (events_context[3, 3 +18] == events[1, 2])
assert (events_context[3, 4 +18] == events[2, 2])

assert (events_context[3, 0 ] == events[0, 3])
assert (events_context[3, 3 ] == events[1, 3])
assert (events_context[3, 4 ] == events[2, 3])



assert (events_context[4, 0 +18] == events[3, 2])
assert (events_context[4, 9 +18] == events[2, 2])

assert (events_context[4, 0] == events[3, 3])
assert (events_context[4, 9] == events[2, 3])


print("TIME SURFACE")
events_context = event_array_patch_time_surface(events_raw, res, dt, 3)

assert (events_context[0] == 0).all()

assert (events_context[1, 3] == 0.9)

assert (events_context[2, 1] == 0.9)
assert (events_context[2, 7] == 1.0)

assert (events_context[3, 1] == 0.9)
assert (events_context[3, 7] == 1.0)
assert (events_context[3, 9] == 1.0)

assert (events_context[4, 0] == 0.9)
assert (events_context[4, 1] == 0.9)

print("EOF")