# OASIS v2 — AI-Powered Behavioral Anomaly Detection for Cybersecurity

This is the **Layers 0-5 implementation**: synthetic data → baseline profiling
→ sequence detection → anomaly classification → impact/risk scoring → LLM
reasoning → analyst dashboard. All six layers are wired end to end.

Companion docs this build follows: `ARCHITECTURE.md` (design), `REPORT_STRUCTURE.md`
(how to write it up), `OASIS-execution-runbook.md` (the original phase plan).

## Quick start

```bash
pip install -r requirements.txt

# Layers 0-3: generate data, train/score/classify/risk-score
python3 data_gen/generate_logs.py   # writes data/sessions.jsonl + data/ground_truth.json
python3 run_pipeline.py             # runs Layers 0-3 + threshold sweep, writes results/*.json

# Layer 4: LLM reasoning (offline template fallback works with no API key;
# export ANTHROPIC_API_KEY=sk-... first for real LLM narratives)
python3 run_reasoning.py

# Layer 5: dashboard
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/
```

Everything upstream of the dashboard is deterministic (seeded), so re-running
`run_pipeline.py` reproduces the same numbers.

## What each layer actually does

| Layer | File | What it is |
|---|---|---|
| 0 — Baseline Profiling | `profiling/baseline_model.py` | Per-entity EWMA over 9 features, population-prior blend for cold-start, decay for concept drift, **plus a dedicated cross-session geo-velocity flag for impossible travel** (kept separate from the averaged feature vector — see Bugs Fixed below) |
| 1 — Detection | `detection/gru_autoencoder.py`, `detection/model.py` | From-scratch NumPy GRU autoencoder (normal-only, reconstruction error) + profiler rarity + cold-start rule fallback + geo-velocity flag, combined via `max()` |
| 2 — Classification | `classification/classifier.py` | RandomForest, trained ONLY on true attack-labeled sessions, applied ONLY to sessions Layer 1 flagged |
| 3 — Risk Scoring | `scoring/risk_engine.py`, `scoring/mitre_map.json`, `scoring/asset_criticality.json` | Transparent arithmetic: `impact_score = w1*anomaly_score + w2*asset_criticality + w3*mitre_severity` |
| 4 — Reasoning | `reasoning/llm_layer.py`, `run_reasoning.py` | Grounded narrative generation from `RiskAssessment` JSON only, with an automated groundedness check (no entity/MITRE ID in output that isn't in the input) |
| 5 — Dashboard | `api/main.py`, `dashboard/index.html` | FastAPI serving a ranked alert queue (sorted by `impact_score`), entity history view, contributing factors, and a recommended-action panel |

Run `run_pipeline.py` and read the printed summary — it's the same numbers
that land in `results/metrics.json`.

## Bugs found and fixed (post-verification pass)

Two correctness issues were found by cross-checking `results/metrics.json`
against the code and the brief's evaluation criteria, and both are now fixed:

1. **`impossible_travel` was being completely missed (support: 0).**
   Root cause: the only geo signal (`is_new_geo`) was one of 9 features
   averaged into `rarity_score`, so a single strong deviation got diluted
   into noise by eight normal-looking features. Fix: added
   `_geo_velocity_flag()` in `profiling/baseline_model.py` — a dedicated,
   non-averaged signal that tracks each entity's `last_geo` /
   `last_session_end` and flags a geo change within an implausible time gap
   (`IMPOSSIBLE_TRAVEL_MAX_GAP_HOURS`, currently a flat 3-hour threshold —
   a real system would use actual distance/plausible-speed, flagged as a
   placeholder). Wired into `detection/model.py::score_all()` via `max()`,
   the same pattern already used for the cold-start rule.

2. **Per-class classification metrics were contaminated by Layer 1's false
   positives.** The classifier has no "normal" class to predict, so every
   false positive Layer 1 flagged got force-assigned to whichever attack
   type it resembled most — dragging down `device_spoofing` and
   `credential_stuffing` precision for reasons that had nothing to do with
   the classifier's actual accuracy. Fix: `eval/metrics.py` now has two
   separate functions — `true_positive_classified_events()` (the real
   Layer 2 evaluation, true attacks only) and `false_positive_misrouting()`
   (a separate diagnostic showing where FPs landed, useful for the report's
   limitations section but no longer polluting the accuracy number).

`run_pipeline.py` also now sweeps `THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45,
0.50, 0.55]` and picks the threshold with the best F1 automatically, printing
the full sweep table before the summary.

## Honest limitations (carry into the report's Section 10 verbatim if useful)

1. **No torch, no network in this sandbox.** `detection/gru_autoencoder.py`
   is a hand-written NumPy GRU with manual backprop-through-time, not
   `torch.nn.GRU`. It's mathematically the same cell, just slower and
   batch-size-1. Swapping in `torch.nn.GRU` + minibatching is a drop-in
   replacement — nothing downstream depends on how reconstruction error is
   computed, only on `.reconstruction_error(session)` existing.
2. **Small sample size per attack class.** Only ~4-8 examples are injected
   per attack type (37 attack sessions total across 7 classes). The
   per-class precision/recall table (`results/metrics.json` →
   `per_class_classification_true_positives_only`) is real, not fabricated,
   but noisy with this few examples per class. More injected examples per
   class would tighten this.
3. **Impossible-travel detection uses a flat time threshold, not real
   geo-distance/speed.** `IMPOSSIBLE_TRAVEL_MAX_GAP_HOURS = 3.0` in
   `profiling/baseline_model.py` treats *any* geo change within 3 hours as
   implausible, regardless of how close the two locations actually are. Good
   enough to catch the injected scenario; not a real plausible-speed model.
4. **Placeholder weights/thresholds**, flagged in code where they occur:
   - `scoring/risk_engine.py::DEFAULT_WEIGHTS` and `ACTION_THRESHOLDS`
   - `detection/model.py`'s combination rule (`max` of recon-error, rarity,
     and geo-velocity) and the z-score-to-[0,1] squashing constant
   - `THRESHOLD_SWEEP` values in `run_pipeline.py` (the sweep picks the best
     of these six, not a globally optimal value)
   None of these are tuned against real incident data — they're defensible
   starting points, not claims of optimality.
5. **Static/batch, not streaming.** Sessions are scored in a single pass
   over a pre-generated file, in timestamp order, to mirror how a streaming
   system *would* process them (the profiler is stateful/incremental on
   purpose). There's no actual message queue / real-time ingestion here.
6. **GRU trained on synthetic sessions only.** No claim of real-world
   generalization — the vocabulary, session shapes, and attack patterns are
   all self-generated.
7. **Layer 4's groundedness check is regex-based, not semantic.** It catches
   hallucinated entity IDs and MITRE IDs specifically, not all forms of
   ungrounded content. Good enough as a demonstrable, real metric for the
   report; not a complete factuality guarantee.
8. **Layer 5's action buttons are visual only.** Acknowledge/Isolate/Escalate
   show a confirmation but have no backend action behind them yet, per the
   brief's own allowance ("doesn't need real backend logic behind the
   button, just needs to visibly represent the decision").

## File map
```
oasis/
├── contracts/schemas.py # every layer's I/O contract
├── data_gen/generate_logs.py # synthetic generator + all 8 attack injectors
├── profiling/baseline_model.py # Layer 0 (+ geo-velocity fix)
├── detection/
│ ├── gru_autoencoder.py # NumPy GRU core
│ └── model.py # Layer 1 wrapper (GRU + rarity + cold-start rule + geo-velocity)
├── classification/classifier.py # Layer 2
├── scoring/
│ ├── mitre_map.json # verified MITRE ATT&CK / ATT&CK-for-ICS IDs
│ ├── asset_criticality.json
│ └── risk_engine.py # Layer 3
├── reasoning/llm_layer.py # Layer 4 (LLM narrative + groundedness check)
├── api/main.py # Layer 5 (FastAPI)
├── dashboard/index.html # Layer 5 (ranked alert queue UI)
├── eval/metrics.py # precision/recall/F1, precision@1%, TP-only confusion
│ # matrix, FP misrouting, cold-start buckets, concept-drift
├── run_pipeline.py # orchestrates Layers 0-3 + threshold sweep
├── run_reasoning.py # orchestrates Layer 4 over results/risk_assessments.json
├── data/ # generated (sessions.jsonl, ground_truth.json, entities.json)
├── results/ # generated (anomaly_events, classified_events,
│ # risk_assessments, incident_narratives, metrics,
│ # worked drift example)
└── requirements.txt



```
## What's left

- **Report** (`REPORT_STRUCTURE.md`, section by section) — real, verified
  metrics are now available for Section 8; the two bugs above are exactly
  the kind of thing that belongs in Section 10 if not fully resolved further.
- **Slide deck**, mirroring the report's section order per
  `REPORT_STRUCTURE.md`'s closing instruction.
- Optional polish: real geo-distance-based impossible-travel check instead
  of the flat time threshold; more injected examples per attack class to
  tighten the per-class precision/recall numbers; wire the dashboard's
  action buttons to something real.