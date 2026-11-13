"""Measure how noisy the provided automatic tracking is, so the DFL data can be degraded to match.

Writes a handful of aggregate numbers (missing-ball rate and burst length, tracked-player rate) to
cache/noise_profile.json. Aggregates only -- nothing frame-level -- and the file stays local
(cache/ is gitignored); anyone with the bundle regenerates it with this script.

    python tools/noise_profile.py --data T:\\Barca
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtah_possession import ingest  # noqa: E402
from mtah_possession.config import GAME, SOURCE  # noqa: E402
from mtah_possession.paths import CACHE_DIR, data_dir, use_challenge_repo  # noqa: E402


def runs_of(mask):
    """Lengths of maximal True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()

    use_challenge_repo()
    from games import half_video_blocks
    trk = ingest.load_tracking(data_dir(args.data), GAME, SOURCE)
    visible = np.isfinite(trk["bx"]) & np.isfinite(trk["by"])
    tracked = np.isfinite(trk["px"]) & np.isfinite(trk["py"])

    miss_runs, vis_runs, n_tracked = [], [], []
    for _, first, last in half_video_blocks(GAME):
        v = visible[first:last + 1]
        miss_runs.append(runs_of(~v))
        vis_runs.append(runs_of(v))
        n_tracked.append(tracked[first:last + 1].sum(axis=1))
    miss_runs, vis_runs = np.concatenate(miss_runs), np.concatenate(vis_runs)
    n_tracked = np.concatenate(n_tracked)

    profile = {
        "ball_miss_frac": float(miss_runs.sum() / (miss_runs.sum() + vis_runs.sum())),
        "ball_miss_run_mean": float(miss_runs.mean()),
        "players_tracked_mean": float(n_tracked.mean()),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "noise_profile.json"), "w") as handle:
        json.dump(profile, handle)
    print("wrote cache/noise_profile.json ({} numbers, aggregates only)".format(len(profile)))


if __name__ == "__main__":
    main()
