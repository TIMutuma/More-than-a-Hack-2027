"""Evidence -> per-frame emission scores for the three states H, A, D.

Two independent log-odds per frame:
  team_ev  > 0 favours home, < 0 favours away   (only meaningful while the ball is alive)
  dead_ev  > 0 favours a dead ball

and emissions  H: +team_ev/2   A: -team_ev/2   D: dead_ev.

A missing ball contributes NO team evidence and NO dead evidence by itself: the decoder then
simply holds its current state, which is what "possession stays with the last team" means.
"""
import numpy as np

from . import tagging


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def team_evidence(F, P):
    d_min = np.fmin(F["d_home"], F["d_away"])
    margin = F["d_away"] - F["d_home"]  # > 0: home player is closer
    strength = np.exp(-np.maximum(d_min - P["ctrl_r0_m"], 0.0) / P["ctrl_decay_m"])
    ball_speed = np.nan_to_num(F["ball_speed"], nan=0.0)
    flight = 1.0 / (1.0 + (ball_speed / P["flight_speed_ms"]) ** 2)
    ev = P["ctrl_weight"] * np.tanh(margin / P["ctrl_margin_scale_m"]) * strength * flight
    ev = np.nan_to_num(ev, nan=0.0)
    width = P["w_width"] * np.tanh(F["width_diff"] / P["width_scale_m"])
    return (ev + np.nan_to_num(width, nan=0.0)).astype(np.float64)


def calibrate_motion(F, graded, fps, P):
    """Threshold on whole-pitch walking speed, from algorithmic pseudo-labels on this match.

    Pseudo-dead: the ball was last seen beyond a line and has not reappeared for a while.
    Pseudo-alive: the ball is moving inside the pitch with a player close to it.
    No human decided any of these frames. Falls back to config values if either set is thin.
    """
    d_min = np.fmin(F["d_home"], F["d_away"])
    motion = F["motion"]
    dead = (graded & F["last_out"] & ~F["visible"]
            & (F["gap"] >= P["calib_dead_gap_s"] * fps) & np.isfinite(motion))
    alive = (graded & F["visible"] & ~F["out_now"] & np.isfinite(motion)
             & (np.nan_to_num(F["ball_speed"], nan=0.0) > P["calib_alive_ball_speed_ms"])
             & (np.nan_to_num(d_min, nan=99.0) < P["calib_alive_dist_m"]))
    if dead.sum() >= P["calib_min_frames"] and alive.sum() >= P["calib_min_frames"]:
        md, ma = float(np.median(motion[dead])), float(np.median(motion[alive]))
        if ma > md:
            return (md + ma) / 2.0, max((ma - md) / 4.0, 0.05), True
    return P["slow_threshold_ms"], P["slow_scale_ms"], False


def dead_evidence(F, fps, threshold, scale, P):
    lost = ~F["visible"] & (F["gap"] >= P["out_gap_min_s"] * fps)
    slow = _sigmoid((threshold - F["motion"]) / scale)
    slow = np.where(np.isfinite(slow), slow, 0.5)
    return (P["dead_bias"]
            + P["w_out"] * F["out_now"]
            + P["w_out_gap"] * (F["last_out"] & lost)
            + P["w_edge_gap"] * (F["last_edge"] & lost)
            + P["w_slow"] * slow
            + P["w_still"] * (F["still_frames"] >= P["still_min_s"] * fps)).astype(np.float64)


def emissions(F, blocks, fps, P, tags=None, names=None):
    """[T,3] scores for H, A, D, plus a summary dict of counts and flags (never data values)."""
    T = len(F["visible"])
    graded = np.zeros(T, dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    team_ev = team_evidence(F, P)
    threshold, scale, calibrated = calibrate_motion(F, graded, fps, P)
    dead_ev = dead_evidence(F, fps, threshold, scale, P)

    summary = {"motion_self_calibrated": calibrated}
    if tags is not None and P["use_tagging"]:
        summary.update(tagging.apply(tags, names, F, blocks, fps, team_ev, dead_ev, P))

    emit = np.stack([team_ev / 2.0, -team_ev / 2.0, dead_ev], axis=1)
    return emit, summary
