# Ball possession from automatic tracking: evidence + Viterbi, selected on other games

More than a Hack 2027, Phase 1. FC Barcelona (H) vs Athletic Club (A), 199,319 frames.

## 1. Approach

The ranking metric, `spellF1`, scores **whole possessions** matched by temporal overlap. A
per-frame classifier, however accurate, fragments possessions wherever its evidence flickers. That
is why the reference baseline's per-frame and possession scores disagree so sharply. We therefore
treat the task as **segmentation**:

1. **Evidence.** For every frame we compute two log-odds from the automatic tracking (and, softly,
   the Smart Tagging):
   - `team_ev`: home vs away in possession, valid while the ball is alive;
   - `dead_ev`: ball dead vs alive.
2. **Decoding.** A 3-state hidden-Markov path (H, A, D) is found by Viterbi with explicit
   transition costs.
   - H↔A costs `cost_switch`, so a change of possession needs *sustained* evidence of control by
     the other team. A touch, a deflection or a lost duel does not flip it. This implements the
     challenge's "controlled possession" rule.
   - alive↔D costs `cost_stop` and `cost_start`, so a stoppage needs sustained evidence too.
   - Where the ball is not tracked, the emissions are flat and the path holds its state, which is
     exactly "possession stays with the last team".
3. **Selection on other games.** The ~25 numbers in `config.py` were selected on public labelled
   matches **from other games**, using the challenge's own scorer, after degrading those matches
   to the noise level of the provided tracking. The provided game's public 5 minutes were used
   only to check that the pipeline is wired correctly.

Why not a learned classifier? The provided game cannot supply training labels (RULES.md). The
other games differ in provider, labelling and noise, so a model with few, interpretable,
physically meaningful parameters transfers more safely than a high-capacity one. We confirm this
on held-out games (§5).

## 2. Assets used and dropped

| asset | used? | how / why not |
|---|---|---|
| Tactical ATD (tracking) | **yes**, core | positions of every track slot and the ball, in metres; slot→team from the metadata |
| Smart Tagging | **yes**, soft | restart windows, kick-offs, phase team hints (§3.3) |
| Tactical homography | no | maps pitch→pixels; only useful with the video |
| Tactical / broadcast video | no | not available to us. Ball and player detection from video was out of scope (the challenge supplies tracking for this reason) |
| Broadcast ATD + homography | no | not frame-aligned to the reference timeline, and a third of it is replays and close-ups. The tactical ATD already covers every frame |

Checks done before relying on the data (scripts in `tools/`, output limited to counts):
- Every raw ATD line is `frame:players:ball` with `x,y,speed` groups. There is no hidden
  ball-state or ball-owner channel.
- Each file has a single `DataFormatSpecification` whose slot order equals the metadata `Player`
  order, so the slot→team mapping is sound.

## 3. The model

### 3.1 Who has the ball (`team_ev`)
- **Ball control.** On frames where the ball is tracked, we take the nearest home and nearest
  away player.
  `team_ev = w · tanh((d_away − d_home)/s) · exp(−max(d_min − r₀, 0)/λ) · 1/(1 + (v_ball/v_f)²)`.
  - The nearer team is favoured in proportion to its margin.
  - Credit decays as even the nearest player gets farther from the ball.
  - Credit fades for a ball in flight, whose nearest player says little.
- **Team width (no ball needed).** The side in possession stretches the pitch; the side without it
  narrows. On the labelled games, the sign of (home width − away width), with width taken as the
  standard deviation of players' y, alone predicts the team in possession on **80–82%** of
  alive frames. Added as `w_width · tanh(Δwidth/s_w)`, it gives the decoder evidence on the 41% of
  frames where the ball is not tracked. It was the single largest improvement in selection.

### 3.2 Whether the ball is dead (`dead_ev`)
- A per-frame prior towards "alive" (`dead_bias`).
- The ball visible beyond a line.
- The ball last seen beyond, or within 1.5 m of, a line, and not seen since.
- **The whole pitch slowing to a walk**: the median player speed, computed from 0.2 s
  displacements and capped at 12 m/s to drop identity switches, then smoothed over ±1 s. Its
  threshold is **self-calibrated on this match** from algorithmic pseudo-labels:
  - pseudo-dead: the ball was last seen out and has been unseen for 2 s;
  - pseudo-alive: the ball is moving inside the pitch with a player close to it.

  No person labels any frame.
- A ball that is visible and motionless for over 1 s, as when placed for a restart.

### 3.3 Smart Tagging (soft)
- **Kick-off (Start):** the graded frames before a half's kick-off are dead. In the second half,
  grading starts before the kick-off.
- **SET PIECES:** the first 40% of the tagged window is pushed towards dead. After it, the tagged
  team is pushed towards possession, because the team taking the restart has the ball.
- **Phase codes** give weak team evidence. Whether a code means "the tagged team has the ball"
  (e.g. BUILD UP) or "the tagged team is defending" (e.g. DEFENDING IN …) is **not assumed**. It
  is read off the tracking: inside each code's intervals, which side is the tagged team mostly on
  when someone is close to the ball? This independently recovered all four DEFENDING /
  DEFENSIVE TRANSITION codes as "defending". `N/A`-tagged intervals, including every throw-in,
  give no team evidence.
- **Not validated.** Tagging exists only for the provided game, so its weights could not be
  selected on other games. They were fixed in advance, below the weakest validated evidence, so
  they can only tip near-ties. They change about 9.5% of frames.

### 3.4 Possession rules, in summary
- **Loose and contested balls** stay with the last team in control, unless the other team's
  control is sustained enough to pay `cost_switch`.
- **Restarts** belong to the team controlling the ball when play resumes, with the set-piece tag
  as a hint.
- **Dead ball** is labelled `D` and never owned (CHALLENGE.md).
- **Abstention.** We output `X` only on ungraded frames. On graded frames, the scorer turns `X`
  into "repeat the previous label", which the decoder already does when evidence is absent.

## 4. Training / selection data (none from the provided game)

| dataset | games | labels | who produced them | licence / source |
|---|---|---|---|---|
| Metrica Sports sample data | 3 (≈ 425k labelled frames, ≈ 1,130 possessions) | per-frame H/A/D **derived by our fixed rules** from the games' event data (below) | event data published by Metrica Sports (anonymised games). The annotation process is Metrica's | github.com/metrica-sports/sample-data |
| IDSSE / DFL open data | 7 Bundesliga matches (≈ 1.0M labelled frames, ≈ 3,060 possessions) | per-frame `BallStatus` + `BallPossession` from the DFL positional feed | DFL / Sportec Solutions, see Bassek et al. (2025) | CC-BY 4.0, doi:10.1038/s41597-025-04505-y |

**Rules for the Metrica sample labels** (fixed before any scoring; `calib/metrica_sample.py`):
- **Alive, team T** from the start of any ball-playing event of T: SET PIECE, PASS, RECOVERY,
  BALL LOST, SHOT, CARRY. CHALLENGEs do not change possession.
- **Dead** from the end of a BALL OUT, from the end of a shot that goes out or scores, and from a
  FAULT RECEIVED, until the next ball-playing event.

Sanity check: with a player within 1.5 m of the ball, the nearest team agrees with these labels
on 92–94% of frames.

**Matching the noise.** The DFL and Metrica sample tracking is cleaner than the provided ATD, so
every game is degraded before features are computed (`calib/degrade.py`, fixed seeds):
- burst-wise ball drop-outs up to the ATD's measured missing rate and burst length;
- player drop-outs up to the ATD's tracked-player rate;
- positional jitter.

The ATD noise profile is three aggregates, measured by `tools/noise_profile.py`.

**Selection.** A coordinate search maximises the **source-balanced** mean `spellF1`: the mean over
the Metrica games and the mean over the DFL games, averaged. That way the 7 DFL games do not
outvote the 3 Metrica games, which come from the challenge's own provider. `spellF1` is computed
by the challenge's `possession_metrics.py`.

## 5. Results

`spellF1` from the official metric code, on games not used to choose the parameters being scored:

| evaluation | reference baseline (`smooth50`) | this model |
|---|---|---|
| **Pure hold-out.** An earlier configuration was selected on 4 games (3 Metrica + 1 DFL). The other 6 DFL matches finished downloading only afterwards, and were then scored once | 0.184 | **0.332** |
| **Leave-one-game-out** over all 10 games: search on 9, score the 10th, repeat. Source-balanced | TODO | **TODO** |

Every held-out game improved. For scale, the same baseline scores **0.2433** on the provided
match's full private ground truth. Re-selecting on all 10 games raised the source-balanced
in-sample score only from 0.436 to 0.448, so the parameters are not finely tuned to any one game.

On the provided match, the submitted file reports **315 possessions** of mean 12.8 s. The public
window's rate extrapolates to about 360 over the 106 graded minutes. On the public 5 minutes it
scores `spellF1` 0.390 and `macroF1` 0.738, against 0.326 and 0.621 for `smooth50`. That window
has only 17 possessions, and the scorer reports gaps below about 0.33 as noise there, so it served
as a correctness check, not a selection signal.

## 6. Limitations and what we would do next
- **Label definitions differ between sources.** DFL marks roughly twice as many possession
  changes as our Metrica-event rules, and both may differ from Metrica's corrected tracking. The
  model under-predicts possession counts on DFL. With more time we would learn the switch cost
  per source and check which definition the public window resembles. That check is allowed for
  understanding, not tuning.
- **The noise model is simple.** Ball drop-outs are independent of context. In the ATD they
  cluster in crowded boxes and aerial play, exactly where possession changes. A context-dependent
  drop-out model would make selection more faithful.
- **Smart Tagging weights are unvalidated** (§3.3). So are the out-of-play weights, on the Metrica
  games, whose ball is never tracked beyond the lines. Those were calibrated by the DFL games only.
- **Hand-set emission forms.** A learned emission model (logistic or gradient boosting on the
  10 games), pass and carrier persistence, and attacking-direction features are the obvious next
  steps.
- **Video was not used.** Ball detection from the tactical video would directly attack the 41%
  of frames without a ball.

## 7. Reproduction

```bash
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git
git clone <this repository> mtah-possession && cd mtah-possession
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m mtah_possession.run --data <bundle with tactical/> --name <team>
```

Deterministic: two runs give a byte-identical CSV, from a cold cache or a warm one. Parameter
selection is reproducible too: `tools/noise_profile.py`, then `tools/fetch_metrica_sample.py`,
`tools/fetch_idsse.py` and `python -m mtah_possession.calib.cv --search --loo`. See the README.
