"""Where things live. The challenge repository is imported, never copied or modified.

The organisers' repo provides the frame alignment (games.py), the ATD metadata parser
(starter/baseline.py) and -- most importantly -- the official metric and validator
(evaluation/). Using them directly means our local numbers are computed by the same code that
produces the official result.
"""
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
OUT_DIR = os.path.join(REPO_ROOT, "out")


def challenge_repo():
    """Path to a clone of metrica-sports/more-than-a-hack-2027 (MTAH_REPO, else a sibling)."""
    path = os.environ.get("MTAH_REPO") or os.path.join(
        os.path.dirname(REPO_ROOT), "more-than-a-hack-2027")
    if not os.path.isfile(os.path.join(path, "games.py")):
        raise SystemExit(
            "challenge repository not found at {}\n"
            "  git clone https://github.com/metrica-sports/more-than-a-hack-2027.git\n"
            "next to this repository, or set MTAH_REPO.".format(path))
    return path


def use_challenge_repo():
    """Put games.py, starter/ and evaluation/ on sys.path. Idempotent."""
    root = challenge_repo()
    for sub in ("evaluation", "starter", ""):
        path = os.path.join(root, sub) if sub else root
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def data_dir(explicit=None):
    """The match bundle: --data, else MTAH_DATA_DIR, else the challenge repo's data/."""
    path = explicit or os.environ.get("MTAH_DATA_DIR") or os.path.join(challenge_repo(), "data")
    return os.path.abspath(path)
