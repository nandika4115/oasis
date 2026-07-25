"""
OASIS v2 — Layer 5: API
Serves the pipeline output (RiskAssessment + IncidentNarrative merged) and
the dashboard.

Run:
    pip install fastapi uvicorn
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
DASHBOARD_DIR = BASE_DIR / "dashboard"

app = FastAPI(title="OASIS v2 API")


def _load(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    return json.load(open(path))


@app.get("/api/incidents")
def get_incidents():
    """Ranked alert queue: RiskAssessment + IncidentNarrative merged,
    sorted by impact_score descending."""
    risk_assessments = _load("risk_assessments.json")
    narratives = {n["session_id"]: n for n in _load("incident_narratives.json")}
    combined = []
    for r in risk_assessments:
        record = dict(r)
        narrative = narratives.get(r["session_id"])
        if narrative:
            record["narrative_summary"] = narrative.get("summary")
            record["narrative_why_flagged"] = narrative.get("why_flagged")
            record["narrative_confidence"] = narrative.get("confidence")
            record["groundedness_passed"] = narrative.get("_groundedness_check", {}).get("passed")
        combined.append(record)
    combined.sort(key=lambda r: -r["impact_score"])
    return combined


@app.get("/api/metrics")
def get_metrics():
    return _load("metrics.json")


app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/")
def dashboard():
    return FileResponse(str(DASHBOARD_DIR / "index.html"))