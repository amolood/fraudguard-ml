"""FraudGuard — API for ML detection of financial cybercrime in banking transactions."""
from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import sys
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fraudguard import __version__  # noqa: E402
from fraudguard.datasets import DATASETS, ENGINEER, LABEL_COLUMN, RAW_COLUMNS, base_feature, label_ar  # noqa: E402

STATIC, REPORTS = ROOT / "app" / "static", ROOT / "reports"
DB, MODELS, METRICS = ROOT / "data" / "analysis.db", ROOT / "models", ROOT / "models" / "metrics.json"
MAX_UPLOAD_BYTES, MAX_BATCH_ROWS = 50 * 1024 * 1024, 500_000

_lock = threading.Lock()
state: dict = {"models": {}, "metrics": {}}


@asynccontextmanager
async def lifespan(app):
    for key in DATASETS:
        path = MODELS / f"{key}.joblib"
        if path.exists():
            # joblib uses pickle: load only artifacts produced locally by train.py.
            state["models"][key] = joblib.load(path)
    if METRICS.exists():
        state["metrics"] = json.loads(METRICS.read_text(encoding="utf-8"))
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS screenings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, dataset TEXT NOT NULL,
            decision TEXT NOT NULL, probability REAL NOT NULL, score REAL NOT NULL,
            features TEXT NOT NULL, top_reasons TEXT NOT NULL)""")
    yield


app = FastAPI(title="FraudGuard", version=__version__, lifespan=lifespan,
              description="Machine-learning detection of fraud and money laundering in banking transactions")
# Local-only service: reject foreign Host headers (DNS rebinding) and require a custom
# header on state-changing requests (a cross-site form or image cannot set it).
ALLOWED_HOSTS = os.environ.get("FRAUDGUARD_ALLOWED_HOSTS", "127.0.0.1,localhost,testserver").split(",")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
CLIENT_HEADER = "fraudguard-ui"


def same_origin_client(x_requested_with: str | None = Header(default=None)) -> None:
    if x_requested_with != CLIENT_HEADER:
        raise HTTPException(403, f"Missing header X-Requested-With: {CLIENT_HEADER}")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
REPORTS.mkdir(exist_ok=True)
app.mount("/reports", StaticFiles(directory=REPORTS), name="reports")


class Transaction(BaseModel):
    features: dict[str, str | float | int | None] = Field(
        description="Model features of one transaction; missing ones fall back to the training median/mode.")


def get(key: str) -> tuple[dict, dict]:
    if key not in DATASETS:
        raise HTTPException(404, f"Unknown dataset '{key}'. Choose one of {list(DATASETS)}.")
    art, met = state["models"].get(key), state["metrics"].get("datasets", {}).get(key)
    if art is None or met is None:
        raise HTTPException(503, f"Model for '{key}' unavailable. Run python train.py first.")
    return art, met


def decide(score: float, thr: dict) -> str:
    return "block" if score >= thr["block"] else "review" if score >= thr["review"] else "approve"


def build_frame(met: dict, features: dict) -> tuple[pd.DataFrame, list[str]]:
    schema = {f["name"]: f for f in met["schema"]}
    unknown = set(features) - set(schema)
    errors = [f"Unknown fields: {sorted(unknown)}"] if unknown else []
    row, defaulted = {}, []
    for name, spec in schema.items():
        v = features.get(name)
        if v is None or v == "":
            row[name] = spec["default"] if spec["type"] != "number" or spec["default"] is not None else np.nan
            defaulted.append(name)
        elif spec["type"] == "category":
            if str(v) not in spec["options"]:
                errors.append(f"{name}: must be one of {spec['options']}")
            row[name] = str(v)
        else:
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{name}: must be numeric")
                continue
            if not math.isfinite(x):
                errors.append(f"{name}: must be finite")
            elif spec["type"] == "binary" and x not in (0, 1):
                errors.append(f"{name}: must be 0 or 1")
            row[name] = x
    if errors:
        raise HTTPException(422, "; ".join(errors))
    return pd.DataFrame([row]), defaulted


def explain(art: dict, frame: pd.DataFrame, top: int = 8) -> list[dict]:
    ds = DATASETS[art["dataset"]]
    prep, clf = art["model"].named_steps["prep"], art["model"].named_steps["clf"]
    Z = prep.transform(frame[art["features"]])
    contrib = (clf.booster_.predict(Z, pred_contrib=True) if art["algorithm"] == "LightGBM"
               else clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True))[0, :-1]
    agg: dict[str, float] = defaultdict(float)
    for name, c in zip(art["feature_names_out"], contrib):
        agg[base_feature(ds, name.replace("missingindicator_", ""))] += float(c)
    raw = frame.iloc[0].to_dict()
    return [{"feature": f, "label_ar": label_ar(ds, f), "contribution": round(v, 4),
             "value": None if f not in raw or (isinstance(raw[f], float) and math.isnan(raw[f])) else raw[f]}
            for f, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:top]]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "models_loaded": sorted(state["models"])}


@app.get("/api/overview")
def overview():
    with sqlite3.connect(DB) as conn:
        total = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
        by = dict(conn.execute("SELECT decision, COUNT(*) FROM screenings GROUP BY decision").fetchall())
    m = state["metrics"]
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("schema", "samples", "test_runs", "validation_runs")}
            for k, v in m.get("datasets", {}).items()}
    return {"datasets": slim, "cross_dataset": m.get("cross_dataset", {}), "protocol": m.get("protocol", {}),
            "trained_at": m.get("trained_at"), "screenings": total, "decisions": by}


@app.get("/api/schema/{key}")
def schema(key: str):
    art, met = get(key)
    return {"dataset": key, "title_ar": met["title_ar"], "fields": met["schema"], "samples": met["samples"],
            "thresholds": art["thresholds"], "model": art["model_name"]}


@app.post("/api/score/{key}", dependencies=[Depends(same_origin_client)])
def score(key: str, tx: Transaction):
    art, met = get(key)
    frame, defaulted = build_frame(met, tx.features)
    s = float(art["model"].predict_proba(frame[art["features"]])[0, 1])
    p = float(art["calibrator"].predict([s])[0])
    decision = decide(s, art["thresholds"])
    reasons = explain(art, frame)
    now = datetime.now(timezone.utc).isoformat()
    with _lock, sqlite3.connect(DB, timeout=10) as conn:
        ident = conn.execute("INSERT INTO screenings(created_at,dataset,decision,probability,score,features,top_reasons) "
                             "VALUES (?,?,?,?,?,?,?)",
                             (now, key, decision, p, s, json.dumps(frame.iloc[0].to_dict(), default=str),
                              json.dumps(reasons[:3], ensure_ascii=False))).lastrowid
    return {"id": ident, "dataset": key, "created_at": now, "decision": decision, "probability": round(p, 5),
            "score": round(s, 6), "thresholds": art["thresholds"], "reasons": reasons, "defaulted_fields": defaulted,
            "model": art["model_name"], "notice": "Risk estimate to support an analyst; not proof of a crime."}


@app.post("/api/score-batch/{key}", dependencies=[Depends(same_origin_client)])
async def score_batch(key: str, file: UploadFile = File(...)):
    """Raw transactions (original dataset columns) -> behavioural features -> scores."""
    art, _ = get(key)
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File larger than 50 MB.")
    try:
        raw = pd.read_csv(io.BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not parse CSV: {exc}") from exc
    if len(raw) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Maximum {MAX_BATCH_ROWS:,} rows per file.")
    missing = [c for c in RAW_COLUMNS[key] if c not in raw.columns]
    if missing:
        raise HTTPException(422, f"Missing columns: {missing}")
    try:
        df = ENGINEER[key](raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not build features: {exc}") from exc
    if df.empty:
        raise HTTPException(422, "No scorable transactions in the file.")
    s = art["model"].predict_proba(df[art["features"]])[:, 1]
    p = art["calibrator"].predict(s)
    t = art["thresholds"]
    dec = np.where(s >= t["block"], "block", np.where(s >= t["review"], "review", "approve"))
    out = {"rows_in_file": int(len(raw)), "rows_scored": int(len(df)),
           "decisions": {k: int((dec == k).sum()) for k in ("approve", "review", "block")}}
    if LABEL_COLUMN[key] in raw.columns:
        y = df["y"].to_numpy()
        flagged = dec != "approve"
        tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
        out["evaluation"] = {"positives": int(y.sum()), "caught": tp, "false_alarms": fp,
                             "recall": round(tp / max(1, y.sum()), 4), "precision": round(tp / max(1, flagged.sum()), 4),
                             "fpr": round(fp / max(1, (y == 0).sum()), 5)}
    order = np.argsort(-s)[:50]
    has_y = LABEL_COLUMN[key] in raw.columns
    out["top_risk"] = [{"row": int(i), "probability": round(float(p[i]), 5), "decision": str(dec[i]),
                        **({"actual": int(df["y"].iloc[i])} if has_y else {})} for i in order]
    return out


@app.get("/api/history")
def history(limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)):
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT id,created_at,dataset,decision,probability,top_reasons FROM screenings "
                            "ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
    return {"total": total, "items": [{**dict(r), "top_reasons": json.loads(r["top_reasons"])} for r in rows]}


@app.delete("/api/history", dependencies=[Depends(same_origin_client)])
def clear_history():
    with _lock, sqlite3.connect(DB) as conn:
        n = conn.execute("SELECT COUNT(*) FROM screenings").fetchone()[0]
        conn.execute("DELETE FROM screenings")
    return {"deleted": n}
