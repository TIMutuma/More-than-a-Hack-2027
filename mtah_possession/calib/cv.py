"""Select the model's parameters on labelled OTHER games with the OFFICIAL spellF1.

    python -m mtah_possession.calib.cv               # baseline vs current config, per game
    python -m mtah_possession.calib.cv --search      # coordinate search on all games
    python -m mtah_possession.calib.cv --loo         # leave-one-game-out estimate
    python -m mtah_possession.calib.cv --search --loo --workers 6

Uses every fully downloaded game from calib.metrica_sample and calib.dfl. Each is degraded to the
ATD's noise profile first (cache/noise_profile.json, from tools/noise_profile.py). Scores come
from evaluation/possession_metrics.py in the challenge repo, so they mean exactly what the
leaderboard means.

Objective (fixed before any result was seen): the mean spellF1 of each SOURCE, averaged over
sources -- so 7 DFL matches do not outvote 3 Metrica games, and Metrica is the challenge's own
provider. Smart Tagging exists only for the provided game, so it is switched off here and its
weights are not selected by this script.
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .. import decode, features, model
from ..config import PARAMS
from ..paths import CACHE_DIR, use_challenge_repo
from . import degrade, dfl, metrica_sample

FEATURE_VERSION = 2  # bump when features.py or degrade.py changes
FEATURE_DIR = os.path.join(CACHE_DIR, "calib_features")
FEATURE_KEYS = ("out_margin_m", "edge_m", "speed_lag_frames", "speed_cap_ms", "motion_smooth_s",
                "still_speed_ms", "width_smooth_s")

# Searched in this order, each over its listed values, holding the others fixed.
SEARCH_SPACE = {
    "cost_switch": [1, 2, 3, 4.5, 6, 10, 15, 20, 30],
    "cost_stop": [1, 2, 3.5, 5, 10, 15, 25, 40],
    "cost_start": [2, 3.5, 5, 10, 15, 25, 40],
    "ctrl_weight": [0.25, 0.5, 1.0, 2.0, 4.0],
    "ctrl_r0_m": [0.5, 1.0, 1.5],
    "ctrl_decay_m": [0.5, 1.0, 2.0, 4.0],
    "ctrl_margin_scale_m": [0.5, 1.0, 2.0, 4.0],
    "flight_speed_ms": [6.0, 12.0, 25.0, 50.0],
    "w_width": [0.0, 0.02, 0.05, 0.1, 0.2, 0.4],
    "width_scale_m": [1.0, 2.0, 4.0, 8.0],
    "dead_bias": [-3.0, -2.0, -1.5, -1.0, -0.5],
    "w_slow": [0.0, 1.0, 2.0, 3.0, 4.0],
    "w_out": [1.0, 3.0, 6.0, 12.0],
    "w_out_gap": [0.0, 0.75, 1.5, 3.0, 6.0],
    "w_edge_gap": [0.0, 0.35, 0.7, 1.4, 2.8],
    "w_still": [0.0, 0.5, 1.0, 2.0],
}


# -- data --------------------------------------------------------------------------------------

def sources():
    """[(name, loader)] for every fully downloaded labelled game, in a fixed order."""
    out = [("metrica:" + g, lambda g=g: metrica_sample.load_game(g))
           for g in metrica_sample.available()]
    out += [("dfl:" + m, lambda m=m: dfl.load_match(m)) for m in dfl.available()]
    return out


def _feature_path(name, profile, P):
    key = json.dumps([FEATURE_VERSION, profile, {k: P[k] for k in FEATURE_KEYS}], sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(FEATURE_DIR, "{}_{}.npz".format(name.replace(":", "_"), digest))


def prepare(P, only=None):
    """Compute (or reuse) degraded features for every game; returns their names in order."""
    with open(os.path.join(CACHE_DIR, "noise_profile.json")) as handle:
        profile = json.load(handle)
    os.makedirs(FEATURE_DIR, exist_ok=True)
    names = []
    for name, loader in sources():
        if only and name not in only:
            continue
        path = _feature_path(name, profile, P)
        if not os.path.isfile(path):
            t0 = time.time()
            trk, gt, blocks = loader()
            seed = sum(ord(c) for c in name)  # fixed per game, whatever else is loaded
            F = features.build(degrade.degrade(trk, profile, seed=seed), P)
            np.savez(path, gt=gt, blocks=np.array(blocks), **F)
            print("  {:<24} features built ({:.0f}s)".format(name, time.time() - t0), flush=True)
        names.append(name)
    if not names:
        raise SystemExit("no labelled games downloaded yet: run tools/fetch_metrica_sample.py")
    return names


_LOADED = {}


def _load(name, P):
    """Features of one game, loaded once per process."""
    if name not in _LOADED:
        with open(os.path.join(CACHE_DIR, "noise_profile.json")) as handle:
            profile = json.load(handle)
        with np.load(_feature_path(name, profile, P)) as z:
            F = {k: z[k] for k in z.files if k not in ("gt", "blocks")}
            _LOADED[name] = (F, z["gt"], [tuple(int(v) for v in r) for r in z["blocks"]])
    return _LOADED[name]


# -- scoring -----------------------------------------------------------------------------------

def spell_f1(gt, pred):
    use_challenge_repo()
    import possession_metrics as pm
    runs = pm.valid_runs(gt)
    result = pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)
    return result["mean_f1"], result["n_pred"], result["n_gt"]


def predict(F, blocks, P):
    emit, _ = model.emissions(F, blocks, 25, P)
    pred = np.full(len(emit), 3, dtype=np.int8)
    cost = decode.transition_costs(P)
    for _, first, last in blocks:
        pred[first:last + 1] = decode.viterbi(emit[first:last + 1], cost)
    return pred


def baseline_pred(F, blocks):
    """The organisers' reference: nearest player, majority vote +/- 50 frames (baseline.py)."""
    use_challenge_repo()
    import baseline
    codes = np.full(len(F["visible"]), baseline.NOBODY, dtype=np.int8)
    ok = F["visible"] & np.isfinite(F["d_home"]) & np.isfinite(F["d_away"])
    codes[ok] = np.where(F["d_home"][ok] <= F["d_away"][ok], baseline.HOME, baseline.AWAY)
    letters = baseline.smooth(codes, 50)
    pred = np.where(letters == "H", 0, np.where(letters == "A", 1, 2)).astype(np.int8)
    graded = np.zeros(len(pred), dtype=bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True
    pred[~graded] = 3
    return pred


def _score_task(args):
    name, P, which = args
    F, gt, blocks = _load(name, P)
    pred = baseline_pred(F, blocks) if which == "baseline" else predict(F, blocks, P)
    return spell_f1(gt, pred)


def balanced(names, scores):
    """Mean over sources of each source's mean score."""
    by_source = {}
    for name, s in zip(names, scores):
        by_source.setdefault(name.split(":")[0], []).append(s)
    return float(np.mean([np.mean(v) for v in by_source.values()]))


class Scorer:
    def __init__(self, P, workers):
        self.P0 = P
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None

    def per_game(self, names, P, which="model"):
        tasks = [(n, P, which) for n in names]
        mapped = self.pool.map(_score_task, tasks) if self.pool else map(_score_task, tasks)
        return list(mapped)

    def objective(self, names, P):
        return balanced(names, [r[0] for r in self.per_game(names, P)])


def search(scorer, names, P, rounds=3, log=print):
    P = dict(P)
    best = scorer.objective(names, P)
    for r in range(rounds):
        changed = False
        for key, values in SEARCH_SPACE.items():
            for value in values:
                if value == P[key]:
                    continue
                trial = dict(P, **{key: value})
                score = scorer.objective(names, trial)
                if score > best + 1e-9:
                    best, P, changed = score, trial, True
            log("    round {} {:<22} -> {:<6} objective {:.4f}".format(r + 1, key, P[key], best))
        if not changed:
            break
    return P, best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--loo", action="store_true", help="leave-one-game-out estimate")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--only", nargs="+", help="game names as listed, e.g. metrica:Sample_Game_1")
    args = parser.parse_args()

    use_challenge_repo()
    P = dict(PARAMS, use_tagging=False)
    print("preparing labelled games (degraded to the ATD noise profile)", flush=True)
    names = prepare(P, args.only)
    scorer = Scorer(P, args.workers)

    t0 = time.time()
    base = scorer.per_game(names, P, "baseline")
    cur = scorer.per_game(names, P)
    print("\n{:<24} {:>9} {:>9}   {:>12}".format("game", "baseline", "config", "poss pred/gt"))
    for name, b, m in zip(names, base, cur):
        print("{:<24} {:>9.4f} {:>9.4f}   {:>5}/{:<5}".format(name, b[0], m[0], m[1], m[2]))
    print("{:<24} {:>9.4f} {:>9.4f}   (source-balanced, {:.0f}s per pass)".format(
        "objective", balanced(names, [b[0] for b in base]), balanced(names, [m[0] for m in cur]),
        (time.time() - t0) / 2), flush=True)

    if args.search:
        print("\ncoordinate search on all games", flush=True)
        best_P, best = search(scorer, names, P, log=lambda s: print(s, flush=True))
        changed = {k: best_P[k] for k in SEARCH_SPACE if best_P[k] != P[k]}
        print("best objective {:.4f}; changed vs config: {}".format(best, changed))
        final = scorer.per_game(names, best_P)
        for name, m in zip(names, final):
            print("  {:<24} {:.4f}  poss {}/{}".format(name, m[0], m[1], m[2]))
        with open(os.path.join(CACHE_DIR, "calib_best_params.json"), "w") as handle:
            json.dump({k: best_P[k] for k in SEARCH_SPACE}, handle, indent=1)
        print("wrote cache/calib_best_params.json", flush=True)

    if args.loo:
        print("\nleave-one-game-out (search on the others, score the held-out game)", flush=True)
        rows = []
        for i, name in enumerate(names):
            others = names[:i] + names[i + 1:]
            fitted, _ = search(scorer, others, P, log=lambda s: None)
            held = scorer.per_game([name], fitted)[0][0]
            b = base[i][0]
            rows.append((b, held))
            print("  {:<24} baseline {:.4f}   held-out model {:.4f}".format(name, b, held),
                  flush=True)
        print("LOO source-balanced: baseline {:.4f}   model {:.4f}".format(
            balanced(names, [r[0] for r in rows]), balanced(names, [r[1] for r in rows])))


if __name__ == "__main__":
    main()
