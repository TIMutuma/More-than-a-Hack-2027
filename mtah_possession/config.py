"""Every tunable number in the pipeline, in one place.

Rules for this file:
  * no constant may refer to a specific frame or moment of the provided game;
  * values are set from football priors (Stage 1) or fitted on the IDSSE/DFL open data
    (Stage 2) -- never on the public 5-minute ground truth, which is a pipeline check only.

[S2] marks a value selected by `python -m mtah_possession.calib.cv --search` on 10 labelled
other games (Metrica sample games 1-3 + the 7 IDSSE/DFL matches), degraded to the ATD noise
profile; source-balanced spellF1 0.448 vs 0.266 for the reference baseline on the same games.
Unmarked values are priors the search kept or could not reach (Smart Tagging: not selectable).

Units are SI (metres, seconds, m/s). Weights are per-frame log-odds at 25 fps, so a weight of 1.0
held for one second contributes 25 nats of evidence -- compare that with the decoder costs.
"""

GAME = "BAR-ATH"
SOURCE = "tactical"

PARAMS = {
    # -- geometry ----------------------------------------------------------------------------
    "out_margin_m": 0.5,          # ball this far beyond a line counts as out of play
    "edge_m": 1.5,                # ball last seen this close to a line, then lost: maybe out

    # -- control evidence (who is on the ball) -----------------------------------------------
    "ctrl_r0_m": 1.0,             # within this distance the nearest player is fully credited
    "ctrl_decay_m": 0.5,          # [S2] credit decays by e per this many metres beyond r0
    "ctrl_margin_scale_m": 0.5,   # [S2] tanh scale on (nearest away - nearest home) distance
    "ctrl_weight": 1.0,           # per-frame log-odds at full credit
    "flight_speed_ms": 25.0,      # [S2] faster balls are in flight: nearest player says little

    # -- team shape (evidence on every frame, ball or not) -----------------------------------
    "w_width": 0.2,               # [S2] per-frame log-odds; the wider team tends to have the ball
    "width_scale_m": 8.0,         # [S2] tanh scale on (home width - away width), width = std of y
    "width_smooth_s": 1.0,

    # -- player motion -----------------------------------------------------------------------
    "speed_lag_frames": 5,        # speed from displacement over 0.2 s, robust to jitter
    "speed_cap_ms": 12.0,         # faster than any footballer = identity switch; dropped
    "motion_smooth_s": 1.0,       # half-width of the centred mean on median player speed

    # -- dead-ball evidence ------------------------------------------------------------------
    "dead_bias": -1.5,            # [S2] prior per frame: play is usually alive
    "w_out": 12.0,                # [S2] ball visible beyond a line
    "w_out_gap": 1.5,             # [S2] ball last seen beyond a line and not seen since
    "w_edge_gap": 1.4,            # [S2] ball last seen at a line and not seen since
    "out_gap_min_s": 0.4,
    "w_slow": 2.0,                # whole pitch slowed to a walk
    "slow_threshold_ms": 1.3,     # fallbacks; self-calibrated per match from pseudo-labels
    "slow_scale_ms": 0.3,
    "w_still": 0.5,               # ball visible and motionless for a while: placed for a restart
    "still_speed_ms": 0.5,
    "still_min_s": 1.0,

    # -- self-calibration of the motion threshold (algorithmic pseudo-labels, no humans) -----
    "calib_dead_gap_s": 2.0,      # pseudo-dead: ball last seen out, unseen for this long
    "calib_alive_ball_speed_ms": 2.0,   # pseudo-alive: ball moving inside the pitch...
    "calib_alive_dist_m": 3.0,          # ...with a player close to it
    "calib_min_frames": 500,

    # -- Smart Tagging -----------------------------------------------------------------------
    "use_tagging": True,
    "tag_ctrl_dist_m": 2.0,       # a frame "belongs" to the nearest team within this distance
    "tag_sign_min_share": 0.55,   # a code is read as possession / non-possession only if clear
    "tag_team_weight": 0.05,      # soft: phases are not possessions
    "sp_dead_frac": 0.4,          # first part of a set-piece window: ball still dead
    "sp_dead_weight": 0.6,
    "sp_team_weight": 0.3,        # the restarting team has the ball just after the restart
    "sp_team_after_s": 2.0,
    "prekick_dead": 10.0,         # before a half's kick-off the ball is not in play

    # -- decoder (costs in nats) -------------------------------------------------------------
    "cost_switch": 4.5,           # [S2] H <-> A without a stoppage: needs sustained control
    "cost_stop": 10.0,            # [S2] alive -> D
    "cost_start": 15.0,           # D -> alive
}
