"""Smart Tagging as soft evidence. It is automatic and phase-level, so it never decides a label alone.

Two uses:
  * SET PIECES are restarts: the ball was dead just before, and the tagged team (normally the one
    taking it) has the ball just after. Kick Off (Start) also tells us the ball is not in play
    between the start of a graded half and its kick-off.
  * Every other team-tagged phase is weak evidence about which team has the ball. Whether a code
    means "the tagged team has the ball" (BUILD UP) or "the tagged team is defending"
    (DEFENDING IN ...) is NOT assumed: it is read off the tracking, by checking which side the
    tagged team is mostly on when someone is close to the ball inside that code's intervals.
"""
import numpy as np

from .paths import use_challenge_repo

SET_PIECES = "set pieces"
KICK_OFF_START = "kick off (start)"


def load(data_dir, game):
    use_challenge_repo()
    import load_smart_tagging  # challenge repo, starter/
    return load_smart_tagging.load(game, data_dir)


def team_index(value, names):
    """0 home, 1 away, None for N/A or anything unrecognised. Case-insensitive."""
    value = (value or "").strip().lower()
    for index, name in enumerate(names):
        if value and value == str(name).strip().lower():
            return index
    return None


def code_signs(tags, ctrl_team, names, P):
    """{code: +1 tagged team has the ball, -1 tagged team does not, 0 unclear}."""
    agree, total = {}, {}
    for tag in tags:
        side = team_index(tag["tags"].get("Team"), names)
        if side is None:
            continue
        window = ctrl_team[max(tag["start_frame"], 0):max(tag["end_frame"], 0)]
        window = window[window >= 0]
        code = tag["code"].strip().lower()
        agree[code] = agree.get(code, 0) + int(np.sum(window == side))
        total[code] = total.get(code, 0) + len(window)
    signs = {}
    for code, n in total.items():
        share = agree[code] / float(n) if n else 0.5
        signs[code] = (1 if share >= P["tag_sign_min_share"]
                       else -1 if share <= 1 - P["tag_sign_min_share"] else 0)
    return signs


def apply(tags, names, F, blocks, fps, team_ev, dead_ev, P):
    """Add tagging evidence in place. Returns counts for the run summary (no data values)."""
    T = len(team_ev)
    d_min = np.fmin(F["d_home"], F["d_away"])
    close = F["visible"] & (d_min < P["tag_ctrl_dist_m"])
    ctrl_team = np.where(close, np.where(F["d_away"] > F["d_home"], 0, 1), -1)

    signs = code_signs(tags, ctrl_team, names, P)
    sp_sign = signs.get(SET_PIECES, 0) or 1  # unclear -> the usual meaning, the restarting team
    after = int(round(P["sp_team_after_s"] * fps))

    n_phase = n_restart = 0
    for tag in tags:
        start, end = max(tag["start_frame"], 0), min(tag["end_frame"], T)
        if end <= start:
            continue
        code = tag["code"].strip().lower()
        side = team_index(tag["tags"].get("Team"), names)
        direction = 0 if side is None else (1 if side == 0 else -1)  # +1 favours home

        if code == SET_PIECES:
            split = start + int(round(P["sp_dead_frac"] * (end - start)))
            dead_ev[start:split] += P["sp_dead_weight"]
            team_ev[split:min(end + after, T)] += P["sp_team_weight"] * sp_sign * direction
            n_restart += 1
        elif signs.get(code, 0) and direction:
            team_ev[start:end] += P["tag_team_weight"] * signs[code] * direction
            n_phase += 1

    n_prekick = 0
    for _, first, last in blocks:
        kicks = [t["start_frame"] for t in tags
                 if t["code"].strip().lower() == SET_PIECES
                 and (t["tags"].get("Type") or "").strip().lower() == KICK_OFF_START
                 and first <= t["start_frame"] <= last]
        if kicks:
            dead_ev[first:min(kicks)] += P["prekick_dead"]
            n_prekick += 1

    return {
        "codes_possession": sum(1 for s in signs.values() if s > 0),
        "codes_defending": sum(1 for s in signs.values() if s < 0),
        "codes_unclear": sum(1 for s in signs.values() if s == 0),
        "phases_used": n_phase, "restarts_used": n_restart, "kickoffs_found": n_prekick,
    }
