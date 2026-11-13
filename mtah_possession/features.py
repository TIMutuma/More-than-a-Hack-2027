"""Per-frame evidence from tracking arrays. Pure numpy, source-agnostic (ATD or DFL).

Nothing here assumes the ball is visible: every ball feature is NaN or False where it is not,
and "how long since the ball was last seen, and where" is a feature in its own right.
"""
import warnings

import numpy as np


def nan_moving_mean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half_width)
    hi = np.minimum(len(x), i + half_width + 1)
    count = n[hi] - n[lo]
    return np.where(count > 0, (c[hi] - c[lo]) / np.maximum(count, 1), np.nan)


def run_length_so_far(cond):
    """Number of consecutive True values ending at each frame (0 where False)."""
    idx = np.arange(len(cond))
    last_false = np.maximum.accumulate(np.where(~cond, idx, -1))
    return np.where(cond, idx - last_false, 0)


def last_seen_index(visible):
    """Index of the most recent frame (<= t) with the ball visible; -1 if none yet."""
    idx = np.arange(len(visible))
    return np.maximum.accumulate(np.where(visible, idx, -1))


def player_speeds(px, py, fps, lag, cap):
    """[T,S] speed in m/s from displacement over `lag` frames; NaN if unobserved or implausible."""
    v = np.full(px.shape, np.nan, dtype=np.float32)
    dx = px[lag:] - px[:-lag]
    dy = py[lag:] - py[:-lag]
    v[lag:] = np.hypot(dx, dy) * (fps / float(lag))
    v[v > cap] = np.nan
    return v


def build(trk, P):
    """dict of [T] arrays. trk is the dict from ingest.load_tracking (or the DFL equivalent)."""
    px, py, bx, by, team = trk["px"], trk["py"], trk["bx"], trk["by"], trk["team"]
    fps = int(trk["fps"])
    length, width = (float(v) for v in trk["pitch"])
    T = len(bx)

    visible = np.isfinite(bx) & np.isfinite(by)

    # -- ball ------------------------------------------------------------------------------
    ball_speed = np.full(T, np.nan)
    both = visible[1:] & visible[:-1]
    ball_speed[1:][both] = np.hypot(np.diff(bx)[both], np.diff(by)[both]) * fps

    m = P["out_margin_m"]
    out_now = visible & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    edge_dist = np.minimum.reduce([bx, length - bx, by, width - by])
    near_edge = visible & ~out_now & (edge_dist < P["edge_m"])

    seen = last_seen_index(visible)
    has_seen = seen >= 0
    seen_c = np.maximum(seen, 0)
    gap = np.where(has_seen, np.arange(T) - seen, T).astype(np.int64)
    last_out = has_seen & out_now[seen_c]
    last_edge = has_seen & near_edge[seen_c]

    still_frames = run_length_so_far(visible & (np.nan_to_num(ball_speed, nan=99.0)
                                                < P["still_speed_ms"]))

    # -- players ---------------------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows are expected
        dist = np.hypot(px - bx[:, None], py - by[:, None])
        d_home = np.nanmin(np.where(team[None, :] == 0, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(team[None, :] == 1, dist, np.nan), axis=1)
        speeds = player_speeds(px, py, fps, P["speed_lag_frames"], P["speed_cap_ms"])
        median_speed = np.nanmedian(speeds, axis=1)
        # Team width: the side in possession stretches the pitch, the side without it narrows.
        # Available on every frame, ball or no ball -- the evidence the 41% of ball-less frames
        # otherwise lack.
        width_home = np.nanstd(np.where(team[None, :] == 0, py, np.nan), axis=1)
        width_away = np.nanstd(np.where(team[None, :] == 1, py, np.nan), axis=1)
    motion = nan_moving_mean(median_speed, int(round(P["motion_smooth_s"] * fps)))
    n_players = np.sum(np.isfinite(px) & np.isfinite(py), axis=1)
    width_diff = nan_moving_mean(width_home - width_away, int(round(P["width_smooth_s"] * fps)))

    return {
        "visible": visible, "ball_speed": ball_speed,
        "out_now": out_now, "last_out": last_out, "last_edge": last_edge, "gap": gap,
        "still_frames": still_frames,
        "d_home": d_home, "d_away": d_away,
        "motion": motion, "n_players": n_players, "width_diff": width_diff,
    }
