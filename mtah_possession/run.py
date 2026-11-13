"""Match data in, submission CSV out.

    python -m mtah_possession.run --data T:\\Barca --name <team_name>
    python -m mtah_possession.run --data T:\\Barca --name <team_name> --score-public

Prints only counts, flags and scores -- never data values -- so the console output is safe to
share. --score-public scores against the PUBLIC 5-minute ground truth with the official metric:
a pipeline correctness check, not a tuning signal (RULES.md, EVALUATION.md).
"""
import argparse
import os
import time

import numpy as np

from . import decode, features, ingest, model, tagging
from .config import GAME, PARAMS, SOURCE
from .paths import OUT_DIR, challenge_repo, data_dir, use_challenge_repo

LETTERS = np.array(["H", "A", "D", "X"])


def predict(data, game=GAME, source=SOURCE, params=PARAMS, log=print):
    """Return (labels [T] of H/A/D/X, summary dict)."""
    use_challenge_repo()
    from games import half_video_blocks

    t0 = time.time()
    trk = ingest.load_tracking(data, game, source)
    log("tracking loaded          {} frames x {} slots  ({:.0f}s)".format(
        trk["px"].shape[0], trk["px"].shape[1], time.time() - t0))

    F = features.build(trk, params)
    blocks = half_video_blocks(game)
    tags = tagging.load(data, game) if params["use_tagging"] else None
    emit, summary = model.emissions(F, blocks, trk["fps"], params, tags, list(trk["names"]))

    codes = np.full(len(emit), 3, dtype=np.int8)  # X outside the graded halves: never scored
    cost = decode.transition_costs(params)
    for _, first, last in blocks:
        codes[first:last + 1] = decode.viterbi(emit[first:last + 1], cost)
    log("decoded                  ({:.0f}s total)".format(time.time() - t0))
    return LETTERS[codes], summary


def write_csv(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def check(path, score_public):
    """Official validation (same code as the scorer) + aggregate description, optional score."""
    use_challenge_repo()
    import possession_metrics as pm
    from games import GAME as G, GAMES

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, GAMES[G]["n_video_frames"])
    if reason:
        raise SystemExit("INVALID  {}: {}".format(os.path.basename(path), reason))
    codes = pm.label_codes(table)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    alive = int(np.sum((codes == pm.H) | (codes == pm.A)))
    print("valid                    {}".format(os.path.basename(path)))
    print("labels                   " + "  ".join(
        "{} {}".format(n, int(np.sum(codes == i))) for i, n in enumerate(pm.LABELS)))
    print("possessions              {} of mean {:.1f} s".format(
        len(spells), alive / 25.0 / max(len(spells), 1)))

    if score_public:
        gt_path = os.path.join(challenge_repo(), "evaluation", "gt",
                               "BAR-ATH_possession_gt_public.csv")
        gt = pm.label_codes(pm.read_label_csv(gt_path))
        r = pm.evaluate(gt, codes, bootstrap_n=0)
        print("PUBLIC 5-min check       spellF1 {:.4f}  macroF1 {:.4f}  alive/dead F1 {:.4f}  "
              "possessions {} vs {}".format(r["spell_f1_mean"], r["macro_f1"],
                                            r["alive_dead_f1"], r["n_pred_spells"],
                                            r["n_gt_spells"]))
        print("  (pipeline check only -- 17 possessions cannot rank variants)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", help="match bundle (default: MTAH_DATA_DIR)")
    parser.add_argument("--name", required=True, help="team name: writes out/<name>.csv")
    parser.add_argument("--score-public", action="store_true",
                        help="also score against the public 5-minute ground truth")
    args = parser.parse_args()

    labels, summary = predict(data_dir(args.data))
    for key, value in summary.items():
        print("{:<24} {}".format(key, value))
    path = os.path.join(OUT_DIR, "{}.csv".format(args.name))
    write_csv(path, labels)
    check(path, args.score_public)


if __name__ == "__main__":
    main()
