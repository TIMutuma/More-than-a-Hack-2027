"""Metrica Sports' public sample games -> tracking arrays + per-frame H/A/D labels.

Labels are derived from the games' own (human) event data by FIXED rules written before any
scoring, mirroring the challenge definition (CHALLENGE.md):
  * alive, team T  from the start of any ball-playing event of T (SET PIECE, PASS, RECOVERY,
                   BALL LOST, SHOT, CARRY) -- possession stays with the last team that played the ball,
                   and contested moments (CHALLENGE) do not change it;
  * dead           from the end of a BALL OUT, from the end of a shot that goes out or scores,
                   and from a FAULT RECEIVED -- until the next ball-playing event (the restart);
  * ungraded (X)   before a period's first event and after its last.
These are other games, so using their labels is allowed (RULES.md).

Files from tools/fetch_metrica_sample.py: games 1-2 are CSV (home and away files), game 3 is
FIFA EPTS (the challenge's own format) with JSON events.
"""
import csv
import io
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from ..paths import REPO_ROOT

SAMPLE_DIR = os.path.join(REPO_ROOT, "external", "metrica")
GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")
PITCH = (105.0, 68.0)
CACHE_VERSION = 1
H, A, D, X = 0, 1, 2, 3
BALL_EVENTS = {"SET PIECE", "PASS", "RECOVERY", "BALL LOST", "SHOT", "CARRY"}  # CARRY: game 3


# -- events ------------------------------------------------------------------------------------

def _events_csv(path):
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            team = row["Team"].strip().lower()
            out.append({"side": 0 if team == "home" else 1 if team == "away" else None,
                        "type": row["Type"].strip().upper(),
                        "subtype": (row["Subtype"] or "").strip().upper(),
                        "period": int(row["Period"]),
                        "start": int(row["Start Frame"]), "end": int(row["End Frame"])})
    return out


def _events_json(path, home_id):
    with open(path) as handle:
        data = json.load(handle)["data"]
    out = []
    for e in data:
        sub = e.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        team = (e.get("team") or {}).get("id")
        out.append({"side": None if team is None else (0 if team == home_id else 1),
                    "type": e["type"]["name"].strip().upper(),
                    "subtype": "-".join(s["name"] for s in (sub or [])).upper(),
                    "period": int(e["period"]),
                    "start": int(e["start"]["frame"]), "end": int(e["end"]["frame"] or 0)})
    return out


def labels_from_events(events, n_frames):
    """(gt codes [n_frames] indexed by frame-1, blocks) by the fixed rules in the module doc."""
    gt = np.full(n_frames, X, dtype=np.int8)
    blocks = []
    for period in sorted({e["period"] for e in events}):
        ev = [e for e in events if e["period"] == period]
        first = min(e["start"] for e in ev)
        last = max(max(e["start"], e["end"]) for e in ev)
        changes = []  # (frame, order, state) -- state is 0/1 for a team, D for dead
        for order, e in enumerate(ev):
            t, sub = e["type"], e["subtype"]
            if t in BALL_EVENTS and e["side"] is not None:
                changes.append((e["start"], order, e["side"]))
            ends_play = (t == "BALL OUT"
                         or (t == "SHOT" and ("-OUT" in sub or sub.endswith("GOAL")
                                              or "TARGET-GOAL" in sub)))
            if ends_play:
                changes.append((max(e["start"], e["end"]), order, D))
            if t == "FAULT RECEIVED":
                changes.append((e["start"], order, D))
        changes.sort()
        state = D
        cursor = first
        for frame, _, new_state in changes:
            frame = min(max(frame, first), last + 1)
            gt[cursor - 1:frame - 1] = state
            cursor, state = frame, new_state
        gt[cursor - 1:last] = state
        blocks.append((period, first - 1, last - 1))
    return gt, blocks


# -- tracking ----------------------------------------------------------------------------------

def _tracking_csv(path):
    with open(path) as handle:
        names = [next(handle) for _ in range(3)][2].rstrip("\r\n").split(",")
    table = pd.read_csv(path, skiprows=3, header=None, dtype=np.float64).to_numpy()
    players = [i for i, n in enumerate(names) if n.startswith("Player")]
    ball = names.index("Ball")
    return (table[:, 1].astype(np.int64), table[:, players], table[:, [i + 1 for i in players]],
            table[:, ball], table[:, ball + 1])


def _game_csv(game):
    base = os.path.join(SAMPLE_DIR, game, game)
    fh, hx, hy, bx, by = _tracking_csv(base + "_RawTrackingData_Home_Team.csv")
    fa, ax, ay, _, _ = _tracking_csv(base + "_RawTrackingData_Away_Team.csv")
    if not np.array_equal(fh, fa):
        raise ValueError(game + ": home and away tracking frames differ")
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    return (fh, np.hstack([hx, ax]), np.hstack([hy, ay]), bx, by, team,
            _events_csv(base + "_RawEventsData.csv"))


def _game_epts(game):
    """EPTS with substitutions: each DataFormatSpecification maps the raw groups of its own frame
    range to players, so a raw column is a different person in different parts of the match."""
    base = os.path.join(SAMPLE_DIR, game, game)
    root = ET.parse(base + "_metadata.xml").getroot()
    home_id = root.find(".//Teams/Team").get("id")
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [player_of_channel[r] for r in refs[0::2]]))  # x,y per group
    persons = sorted({p for _, _, ps in specs for p in ps})
    slot = {p: i for i, p in enumerate(persons)}

    with open(base + "_tracking.txt") as handle:
        text = handle.read().translate(str.maketrans(":;", ",,"))
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).to_numpy()
    frames = table[:, 0].astype(np.int64)
    T = len(frames)
    px, py = np.full((T, len(persons)), np.nan), np.full((T, len(persons)), np.nan)
    n_groups = None
    for start, end, ps in specs:
        rows = (frames >= start) & (frames <= end)
        n_groups = len(ps)
        for j, person in enumerate(ps):
            px[rows, slot[person]] = table[rows, 1 + 2 * j]
            py[rows, slot[person]] = table[rows, 2 + 2 * j]
    team = np.array([0 if team_of[p] == home_id else 1 for p in persons], dtype=np.int8)
    return (frames, px, py, table[:, 1 + 2 * n_groups], table[:, 2 + 2 * n_groups], team,
            _events_json(base + "_events.json", home_id))


def load_game(game, use_cache=True):
    """(trk dict, gt codes [T], blocks) for one sample game."""
    cache = os.path.join(SAMPLE_DIR, "{}_v{}.npz".format(game, CACHE_VERSION))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            trk = {k: z[k] for k in ("px", "py", "bx", "by", "team", "pitch", "names")}
            trk["fps"] = 25
            return trk, z["gt"], [tuple(int(v) for v in row) for row in z["blocks"]]

    frames, px, py, bx, by, team, events = (_game_epts(game) if game.endswith("3")
                                            else _game_csv(game))
    if frames[0] != 1 or not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(game + ": tracking frames are not 1..N")
    gt, blocks = labels_from_events(events, len(frames))

    length, width = PITCH
    trk = {"px": (px * length).astype(np.float32), "py": (py * width).astype(np.float32),
           "bx": (bx * length).astype(np.float32), "by": (by * width).astype(np.float32),
           "team": team, "pitch": np.array(PITCH), "names": np.array(["home", "away"]),
           "fps": 25}
    np.savez(cache, gt=gt, blocks=np.array(blocks),
             **{k: v for k, v in trk.items() if k != "fps"})
    return trk, gt, blocks


def available():
    """Games whose files are fully downloaded (tools/fetch_metrica_sample.py)."""
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from fetch_metrica_sample import FILES
    done = []
    for game in GAMES:
        parts = [(rel, size) for rel, size in FILES.items() if rel.startswith(game + "/")]
        if all(os.path.isfile(os.path.join(SAMPLE_DIR, rel))
               and os.path.getsize(os.path.join(SAMPLE_DIR, rel)) == size for rel, size in parts):
            done.append(game)
    return done
