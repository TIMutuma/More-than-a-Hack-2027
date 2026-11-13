"""Stage 2: parameter selection on labelled OTHER games (explicitly allowed by RULES.md).

Sources, each loaded into the same layout as ingest.load_tracking plus per-frame H/A/D codes:
  * metrica_sample -- Metrica Sports' public sample games (same provider as the challenge);
                      labels derived from their event data by fixed rules (metrica_sample.py)
  * dfl            -- IDSSE / DFL open data, labels from BallStatus + BallPossession (optional:
                      2.6 GB from a slow host)
No label from the provided game is used anywhere.
"""
