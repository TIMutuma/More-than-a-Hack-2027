# mtah-possession

Frame-level ball possession (`H` / `A` / `D`) for **More than a Hack 2027 — Phase 1**
(FC Barcelona vs Athletic Club, 199,319 frames), ranked on possession-level `spellF1`.

The approach is in [TECHNICAL.md](TECHNICAL.md). In short:
1. Per-frame evidence from the automatic tracking: who is near the ball, whether play has stopped,
   and soft hints from the Smart Tagging.
2. A three-state Viterbi decoder that turns that evidence into whole possessions.
3. Parameters selected on public labelled data from *other* games, with the official metric.

## Reproduce the submission

Requires Python 3.9+ (built and tested on 3.13), the match bundle, and a clone of the challenge
repository next to this one. The challenge repo supplies the frame alignment, the ATD metadata
parser and the official metric, which are imported, not copied.

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
git clone https://github.com/TIMutuma/More-than-a-Hack-2027.git mtah-possession
cd mtah-possession
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows (bin/ on macOS/Linux)

# the match bundle: a folder holding tactical/ (and optionally broadcast/)
.venv/Scripts/python -m mtah_possession.run --data /path/to/bundle --name <team_name>
```

Clone this repository into a folder **not** named `more-than-a-hack-2027`, for example
`mtah-possession` as above. On case-insensitive file systems (Windows, macOS) its GitHub name
would otherwise collide with the challenge clone next to it.

The submitted file was produced with `--name Timwork`.

This writes `out/<team_name>.csv` and validates it with the official checks. The run is
deterministic: the same inputs give a byte-identical file. The first run parses the tracking into
`cache/`, which takes about 15 s; later runs take about 1 s.

Optional flags and variables:
- `--score-public` also scores the file against the public 5-minute ground truth. This is a
  pipeline check only.
- `MTAH_DATA_DIR` can replace `--data`.
- `MTAH_REPO` points at the challenge clone if it isn't a sibling folder.

## Reproduce the parameter selection (optional)

The parameters in `mtah_possession/config.py` were selected on Metrica Sports' public sample
games. Labels come from those games' event data, so none come from the provided game.

```bash
.venv/Scripts/python tools/noise_profile.py --data /path/to/bundle   # ATD noise, aggregates only
.venv/Scripts/python tools/fetch_metrica_sample.py                   # ~180 MB, public sample data
.venv/Scripts/python -m mtah_possession.calib.cv --search --loo
```

`tools/fetch_idsse.py` adds the IDSSE/DFL open data (7 Bundesliga matches, 2.6 GB) as further
calibration games when downloaded.

## Layout

```
mtah_possession/
  run.py          CLI: data in, validated CSV out
  ingest.py       raw ATD -> dense arrays (cached locally)
  features.py     per-frame evidence (ball, players, motion)
  tagging.py      Smart Tagging -> restart markers, team hints
  model.py        evidence -> H/A/D emission scores
  decode.py       3-state Viterbi
  config.py       every parameter, in one place
  calib/          parameter selection on other games (Metrica samples, IDSSE/DFL)
tools/            data probes and downloaders
```

## Data

The match data is licensed for the competition only and is **not** in this repository.
`.gitignore` excludes the bundle, everything derived from it (`cache/`, `out/`), and the
downloaded open data (`external/`).
