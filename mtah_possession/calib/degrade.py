"""Make clean DFL tracking look like the provided automatic tracking.

DFL positions are corrected and complete; the challenge ATD loses the ball on ~41% of graded
frames, in bursts, and drops players. Parameters a model is selected on clean data would not
transfer, so every DFL match is degraded with the noise profile measured on the ATD
(tools/noise_profile.py) before any feature is computed. Seeded: the same input gives the same
output, every run.
"""
import numpy as np

PLAYER_MISS_RUN = 50      # frames; ATD player drop-outs are tracklet gaps of a couple of seconds
PLAYER_JITTER_M = 0.15
BALL_JITTER_M = 0.20


def markov_missing(T, miss_frac, miss_run_mean, rng):
    """Boolean [T]: True where missing. Alternating geometric runs with the given rate and burst."""
    miss_frac = min(max(miss_frac, 1e-3), 0.95)
    vis_run_mean = miss_run_mean * (1.0 - miss_frac) / miss_frac
    out = np.zeros(T, dtype=bool)
    t = 0
    missing = rng.random() < miss_frac
    while t < T:
        mean = miss_run_mean if missing else vis_run_mean
        length = int(rng.geometric(1.0 / max(mean, 1.0)))
        if missing:
            out[t:t + length] = True
        t += length
        missing = not missing
    return out


def degrade(trk, profile, seed):
    rng = np.random.default_rng(seed)
    T, S = trk["px"].shape
    out = dict(trk)

    # Add only the shortfall: some sources already lose the ball (Metrica's samples do on about a
    # third of frames), and stacking the full ATD rate on top would over-degrade them.
    already = float(np.mean(~(np.isfinite(trk["bx"]) & np.isfinite(trk["by"]))))
    extra = 1.0 - (1.0 - profile["ball_miss_frac"]) / max(1.0 - already, 1e-6)
    ball_missing = (markov_missing(T, extra, profile["ball_miss_run_mean"], rng) if extra > 0
                    else np.zeros(T, dtype=bool))
    bx = trk["bx"] + rng.normal(0, BALL_JITTER_M, T).astype(np.float32)
    by = trk["by"] + rng.normal(0, BALL_JITTER_M, T).astype(np.float32)
    bx[ball_missing], by[ball_missing] = np.nan, np.nan
    out["bx"], out["by"] = bx, by

    on_pitch = float(np.mean(np.sum(np.isfinite(trk["px"]), axis=1)))  # same idea for players
    player_miss = max(0.0, 1.0 - profile["players_tracked_mean"] / max(on_pitch, 1.0))
    px = trk["px"] + rng.normal(0, PLAYER_JITTER_M, (T, S)).astype(np.float32)
    py = trk["py"] + rng.normal(0, PLAYER_JITTER_M, (T, S)).astype(np.float32)
    if player_miss > 0:
        for slot in range(S):
            gone = markov_missing(T, player_miss, PLAYER_MISS_RUN, rng)
            px[gone, slot], py[gone, slot] = np.nan, np.nan
    out["px"], out["py"] = px, py
    return out
