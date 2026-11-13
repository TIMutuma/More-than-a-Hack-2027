"""Read the raw ATD once into dense arrays, cached locally (cache/ is gitignored: never publish it).

Raw line format, verified by tools/probe_format.py on both recordings:
    <frame_id>:<x,y,speed>;<x,y,speed>;...:<x,y,speed>
one player group per track slot, then the ball; NaN where unobserved; coordinates normalised 0-1.
Swapping ':' and ';' for ',' turns the whole file into a plain numeric CSV, which pandas' C parser
reads far faster than a Python loop.

The returned arrays are in metres with the origin at a pitch corner, which is all the features
need. The same dict layout is produced for the IDSSE/DFL matches in Stage 2, so everything
downstream is source-agnostic.
"""
import io
import os

import numpy as np
import pandas as pd

from .paths import CACHE_DIR, use_challenge_repo

CACHE_VERSION = 1


def load_tracking(data_dir, game, source="tactical", use_cache=True):
    """{px, py: [T,S] m; bx, by: [T] m; team: [S] 0 home / 1 away; fps; pitch; names}."""
    use_challenge_repo()
    from games import GAMES, require, source_frames  # challenge repo
    from baseline import team_of_each_track, team_names  # challenge repo, starter/

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    stat = os.stat(raw_path)
    cache = os.path.join(CACHE_DIR, "{}_{}_v{}_{}.npz".format(
        game, source, CACHE_VERSION, stat.st_size))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as z:
            return _as_dict(z)

    with open(raw_path) as handle:
        text = handle.read().translate(str.maketrans(":;", ",,"))
    table = pd.read_csv(io.StringIO(text), header=None, engine="c", dtype=np.float64).to_numpy()
    del text

    n_frames = source_frames(source, game)
    if table.shape[0] != n_frames:
        raise ValueError("{} records, expected {}".format(table.shape[0], n_frames))
    if not np.array_equal(table[:, 0], np.arange(1, n_frames + 1)):
        raise ValueError("frame ids are not 1..{} in order".format(n_frames))
    n_slots, rest = divmod(table.shape[1] - 4, 3)
    if rest:
        raise ValueError("unexpected column count {}".format(table.shape[1]))

    team = team_of_each_track(meta_path)
    if len(team) != n_slots:
        raise ValueError("{} slots in data, {} in metadata".format(n_slots, len(team)))

    length, width = GAMES[game]["pitch"]
    players = table[:, 1:1 + 3 * n_slots]
    ball = table[:, 1 + 3 * n_slots:]
    out = {
        "px": (players[:, 0::3] * length).astype(np.float32),
        "py": (players[:, 1::3] * width).astype(np.float32),
        "bx": (ball[:, 0] * length).astype(np.float32),
        "by": (ball[:, 1] * width).astype(np.float32),
        "team": team.astype(np.int8),
        "fps": np.int64(GAMES[game]["fps"]),
        "pitch": np.array([length, width], dtype=np.float64),
        "names": np.array(team_names(meta_path)),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, **out)
    return out


def _as_dict(z):
    out = {key: z[key] for key in z.files}
    out["fps"] = int(out["fps"])
    return out
