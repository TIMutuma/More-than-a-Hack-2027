"""Parse one IDSSE match into the same dict layout ingest.load_tracking produces, plus labels.

DFL positions XML: one <FrameSet GameSection=.. TeamId=.. PersonId=..> per object per half, each
holding <Frame N=.. X=.. Y=.. [BallStatus=.. BallPossession=..]/>. The ball FrameSet (TeamId
"BALL") carries BallStatus (1 alive) and BallPossession (1 home, 2 guest) -- our ground truth.
Streamed with the standard library and cached as .npz under external/ (gitignored).
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

from ..paths import REPO_ROOT

IDSSE_DIR = os.path.join(REPO_ROOT, "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
SECTIONS = ("firstHalf", "secondHalf")
CACHE_VERSION = 1
H, A, D, X = 0, 1, 2, 3


def _meta(path):
    root = ET.parse(path).getroot()
    env = root.find(".//Environment")
    length, width = float(env.get("PitchX")), float(env.get("PitchY"))
    team_of_person = {}
    for team in root.iter("Team"):
        side = 0 if team.get("Role") == "home" else 1
        for player in team.iter("Player"):
            team_of_person[player.get("PersonId")] = side
    return length, width, team_of_person


def _framesets(path):
    """{(section, object_id): (N, X, Y, status, possession)} as numpy arrays."""
    out = {}
    current, rows = None, None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            obj = "BALL" if elem.get("TeamId", "").upper() == "BALL" else elem.get("PersonId")
            current, rows = (elem.get("GameSection"), obj), []
        elif event == "end" and elem.tag == "Frame":
            g = elem.get
            rows.append((int(g("N")), float(g("X")), float(g("Y")),
                         int(g("BallStatus", "0")), int(g("BallPossession", "0"))))
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            if current[0] in SECTIONS and rows:
                a = np.array(rows, dtype=np.float64)
                out[current] = (a[:, 0].astype(np.int64), a[:, 1], a[:, 2],
                                a[:, 3].astype(np.int8), a[:, 4].astype(np.int8))
            elem.clear()
    return out


def available():
    """Matches whose files are fully downloaded (tools/fetch_idsse.py)."""
    import sys
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from fetch_idsse import FILES
    out = []
    for match, ((_, meta_size), (_, pos_size)) in sorted(FILES.items()):
        meta = os.path.join(IDSSE_DIR, match + "_meta.xml")
        pos = os.path.join(IDSSE_DIR, match + "_positions.xml")
        if (os.path.isfile(meta) and os.path.getsize(meta) == meta_size
                and os.path.isfile(pos) and os.path.getsize(pos) == pos_size):
            out.append(match)
    return out


def load_match(match_id, use_cache=True):
    """(trk dict, gt codes [T], blocks [(half, first, last)]) for one IDSSE match."""
    cache = os.path.join(IDSSE_DIR, "{}_v{}.npz".format(match_id, CACHE_VERSION))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            trk = {k: z[k] for k in ("px", "py", "bx", "by", "team", "pitch", "names")}
            trk["fps"] = 25
            blocks = [tuple(int(v) for v in row) for row in z["blocks"]]
            return trk, z["gt"], blocks

    length, width, team_of_person = _meta(os.path.join(IDSSE_DIR, match_id + "_meta.xml"))
    sets = _framesets(os.path.join(IDSSE_DIR, match_id + "_positions.xml"))
    persons = sorted({obj for (_, obj) in sets if obj != "BALL" and obj in team_of_person})

    px_parts, py_parts, bx_parts, by_parts, gt_parts, blocks = [], [], [], [], [], []
    offset = 0
    for half, section in enumerate(SECTIONS, start=1):
        if (section, "BALL") not in sets:
            continue
        n, x, y, status, poss = sets[(section, "BALL")]
        first = int(n.min())
        T = int(n.max()) - first + 1
        bx, by = np.full(T, np.nan), np.full(T, np.nan)
        gt = np.full(T, X, dtype=np.int8)
        bx[n - first], by[n - first] = x, y
        gt[n - first] = np.where(status == 1, np.where(poss == 1, H, A), D)
        px, py = np.full((T, len(persons)), np.nan), np.full((T, len(persons)), np.nan)
        for slot, person in enumerate(persons):
            if (section, person) in sets:
                pn, pxs, pys, _, _ = sets[(section, person)]
                keep = (pn >= first) & (pn < first + T)
                px[pn[keep] - first, slot], py[pn[keep] - first, slot] = pxs[keep], pys[keep]
        px_parts.append(px); py_parts.append(py); bx_parts.append(bx); by_parts.append(by)
        gt_parts.append(gt)
        blocks.append((half, offset, offset + T - 1))
        offset += T

    px, py = np.concatenate(px_parts), np.concatenate(py_parts)
    bx, by = np.concatenate(bx_parts), np.concatenate(by_parts)
    # DFL's origin is the pitch centre; ours is a corner. Detect rather than assume.
    if np.nanmin(px) < -5 or np.nanmin(py) < -5:
        px, bx = px + length / 2.0, bx + length / 2.0
        py, by = py + width / 2.0, by + width / 2.0

    trk = {
        "px": px.astype(np.float32), "py": py.astype(np.float32),
        "bx": bx.astype(np.float32), "by": by.astype(np.float32),
        "team": np.array([team_of_person[p] for p in persons], dtype=np.int8),
        "pitch": np.array([length, width]), "names": np.array(["home", "guest"]), "fps": 25,
    }
    gt = np.concatenate(gt_parts)
    np.savez(cache, gt=gt, blocks=np.array(blocks), **{k: v for k, v in trk.items() if k != "fps"})
    return trk, gt, blocks
