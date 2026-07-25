# OASIS v2 — AI-Powered Behavioral Anomaly Detection for Cybersecurity

This is the **Layers 0-3 implementation**: synthetic data → baseline profiling
→ sequence detection → anomaly classification → impact/risk scoring. Layers 4
(LLM reasoning/narrative) and 5 (dashboard) are **not built yet** — this repo
stops exactly at `RiskAssessment`, which is the contract Layer 4 is designed
to consume next (see `contracts/schemas.py::IncidentNarrative`, already
stubbed to match).

Companion docs this build follows: `ARCHITECTURE.md` (design), `REPORT_STRUCTURE.md`
(how to write it up), `OASIS-execution-runbook.md` (the original phase plan).

## Quick start

```bash
pip install -r requirements.txt
python3 data_gen/generate_logs.py   # writes data/sessions.jsonl + data/ground_truth.json
python3 run_pipeline.py             # runs Layers 0-3, writes results/*.json, prints metrics
```

Everything is deterministic (seeded), so re-running reproduces the same numbers.

## What each layer actually does

| Layer | File | What it is |
|---|---|---|
| 0 — Baseline Profiling | `profiling/baseline_model.py` | Per-entity EWMA over 9 features, population-prior blend for cold-start, decay for concept drift |
| 1 — Detection | `detection/gru_autoencoder.py`, `detection/model.py` | From-scratch NumPy GRU autoencoder (normal-only, reconstruction error) + profiler rarity + cold-start rule fallback, combined via `max()` |
| 2 — Classification | `classification/classifier.py` | RandomForest, trained ONLY on true attack-labeled sessions, applied ONLY to sessions Layer 1 flagged |
| 3 — Risk Scoring | `scoring/risk_engine.py`, `scoring/mitre_map.json`, `scoring/asset_criticality.json` | Transparent arithmetic: `impact_score = w1*anomaly_score + w2*asset_criticality + w3*mitre_severity` |

Run `run_pipeline.py` and read the printed summary — it's the same numbers
that land in `results/metrics.json`.

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
   `per_class_classification`) is real, not fabricated, but noisy —
   `impossible_travel` and `device_spoofing` in particular have few enough
   examples that a handful of misses swing the numbers a lot. More injected
   examples per class would tighten this.
3. **Placeholder weights/thresholds**, flagged in code where they occur:
   - `scoring/risk_engine.py::DEFAULT_WEIGHTS` and `ACTION_THRESHOLDS`
   - `detection/model.py`'s combination rule (`max` of recon-error and
     rarity) and the z-score-to-[0,1] squashing constant
   - `FLAG_THRESHOLD = 0.45` in `run_pipeline.py`
   None of these are tuned against real incident data — they're defensible
   starting points, not claims of optimality.
4. **Static/batch, not streaming.** Sessions are scored in a single pass
   over a pre-generated file, in timestamp order, to mirror how a streaming
   system *would* process them (the profiler is stateful/incremental on
   purpose). There's no actual message queue / real-time ingestion here.
5. **GRU trained on synthetic sessions only.** No claim of real-world
   generalization — the vocabulary, session shapes, and attack patterns are
   all self-generated.

## What's next (Layers 4-5, not started)

- `reasoning/llm_layer.py` — take `RiskAssessment` → call an LLM with the
  exact JSON as grounded context → parse into `IncidentNarrative` → run the
  groundedness check (no entity/asset name in the output that isn't in the
  input JSON).
- `api/main.py` (FastAPI) + a dashboard — ranked alert queue, entity history,
  contributing factors, recommended-action button. `results/risk_assessments.json`
  (already sorted by `impact_score` descending) is exactly what that queue
  should render.

## File map

```
oasis/
├── contracts/schemas.py          # every layer's I/O contract
├── data_gen/generate_logs.py     # synthetic generator + all 8 attack injectors
├── profiling/baseline_model.py   # Layer 0
├── detection/
│   ├── gru_autoencoder.py        # NumPy GRU core
│   └── model.py                  # Layer 1 wrapper (GRU + rarity + cold-start rule)
├── classification/classifier.py  # Layer 2
├── scoring/
│   ├── mitre_map.json            # verified MITRE ATT&CK / ATT&CK-for-ICS IDs
│   ├── asset_criticality.json
│   └── risk_engine.py            # Layer 3
├── eval/metrics.py               # precision/recall/F1, precision@1%, confusion matrix,
│                                  # cold-start buckets, concept-drift before/after
├── run_pipeline.py               # orchestrates Layers 0-3 end to end
├── data/                         # generated (sessions.jsonl, ground_truth.json, entities.json)
├── results/                      # generated (anomaly_events, classified_events,
│                                  # risk_assessments, metrics, worked drift example)
└── requirements.txt
```
