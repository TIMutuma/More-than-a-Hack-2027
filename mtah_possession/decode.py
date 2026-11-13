"""Three-state Viterbi: the whole-possession structure the metric rewards.

States 0 = H, 1 = A, 2 = D (the same codes as the official possession_metrics). The transition
costs are what turn flickering per-frame evidence into possessions:
  * H <-> A costs `cost_switch`: a change of possession needs sustained evidence of control by the
    other team, so a touch, a deflection or a lost duel does not flip it;
  * alive -> D and D -> alive cost `cost_stop` / `cost_start`: stoppages need sustained evidence too.
Through a ball-missing stretch the emissions are flat and the path simply holds its state.
"""
import numpy as np


def transition_costs(P):
    s, stop, start = P["cost_switch"], P["cost_stop"], P["cost_start"]
    return np.array([[0.0, s, stop],
                     [s, 0.0, stop],
                     [start, start, 0.0]])


def viterbi(emit, cost):
    """Most likely state path for emissions [T,3] under transition costs [3,3] (from, to).

    Plain-Python inner loop: on 3 states it is several times faster than per-step numpy, and the
    calibration search runs it thousands of times. Ties go to the lowest state index, so the path
    is deterministic.
    """
    T = len(emit)
    rows = np.asarray(emit, dtype=np.float64).tolist()
    (c00, c01, c02), (c10, c11, c12), (c20, c21, c22) = np.asarray(cost, dtype=np.float64).tolist()
    back = bytearray(3 * T)
    s0, s1, s2 = rows[0]
    for t in range(1, T):
        e0, e1, e2 = rows[t]
        a, b, c = s0 - c00, s1 - c10, s2 - c20
        if a >= b and a >= c:
            n0, k0 = a, 0
        elif b >= c:
            n0, k0 = b, 1
        else:
            n0, k0 = c, 2
        a, b, c = s0 - c01, s1 - c11, s2 - c21
        if a >= b and a >= c:
            n1, k1 = a, 0
        elif b >= c:
            n1, k1 = b, 1
        else:
            n1, k1 = c, 2
        a, b, c = s0 - c02, s1 - c12, s2 - c22
        if a >= b and a >= c:
            n2, k2 = a, 0
        elif b >= c:
            n2, k2 = b, 1
        else:
            n2, k2 = c, 2
        i = 3 * t
        back[i], back[i + 1], back[i + 2] = k0, k1, k2
        s0, s1, s2 = n0 + e0, n1 + e1, n2 + e2
    path = np.empty(T, dtype=np.int8)
    state = 0 if (s0 >= s1 and s0 >= s2) else (1 if s1 >= s2 else 2)
    path[-1] = state
    for t in range(T - 1, 0, -1):
        state = back[3 * t + state]
        path[t - 1] = state
    return path
