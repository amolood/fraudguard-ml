import io
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
KEYS = list(METRICS["datasets"])


def test_health_and_overview():
    with TestClient(app, headers={"X-Requested-With": "fraudguard-ui"}) as c:
        assert c.get("/").status_code == 200
        assert sorted(c.get("/api/health").json()["models_loaded"]) == sorted(KEYS)
        ov = c.get("/api/overview").json()
        assert set(ov["datasets"]) == set(KEYS) and "schema" not in ov["datasets"][KEYS[0]]
        assert c.get("/api/schema/unknown").status_code == 404


@pytest.mark.parametrize("key", KEYS)
def test_protocol_and_quality(key):
    m = METRICS["datasets"][key]
    assert m["final_test"]["pr_auc"] > 10 * m["split"]["test"]["prevalence"]  # far better than chance
    assert all(not f.startswith("err_balance") for f in m["features_used"])     # simulator artefact excluded


@pytest.mark.parametrize("key", KEYS)
def test_score_samples_and_validation(key):
    with TestClient(app, headers={"X-Requested-With": "fraudguard-ui"}) as c:
        s = c.get(f"/api/schema/{key}").json()
        assert s["thresholds"]["block"] >= s["thresholds"]["review"]
        for sample in s["samples"]:
            r = c.post(f"/api/score/{key}", json={"features": sample["features"]})
            assert r.status_code == 200, r.text
            body = r.json()
            assert body["decision"] in {"approve", "review", "block"} and 0 <= body["probability"] <= 1
            assert body["reasons"]
        high = next(x for x in s["samples"] if "عالية" in x["label"])
        assert c.post(f"/api/score/{key}", json={"features": high["features"]}).json()["decision"] == "block"
        assert c.post(f"/api/score/{key}", json={"features": {"bogus": 1}}).status_code == 422


def test_batch_raw_paysim_and_history():
    if "paysim" not in KEYS:
        pytest.skip("paysim not trained")
    raw = pd.DataFrame({"step": [1, 1, 2], "type": ["TRANSFER", "CASH_OUT", "PAYMENT"], "amount": [9000, 9000, 10],
                        "nameOrig": ["a", "b", "c"], "oldbalanceOrg": [9000, 9000, 50], "newbalanceOrig": [0, 0, 40],
                        "nameDest": ["x", "y", "m"], "oldbalanceDest": [0, 0, 0], "newbalanceDest": [0, 9000, 0],
                        "isFraud": [1, 1, 0]})
    with TestClient(app, headers={"X-Requested-With": "fraudguard-ui"}) as c:
        r = c.post("/api/score-batch/paysim", files={"file": ("b.csv", io.BytesIO(raw.to_csv(index=False).encode()), "text/csv")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rows_in_file"] == 3 and body["rows_scored"] == 2 and "evaluation" in body
        bad = c.post("/api/score-batch/paysim", files={"file": ("b.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
        assert bad.status_code == 422
        assert c.get("/api/history").json()["total"] >= 1
        assert c.delete("/api/history").status_code == 200


def test_state_changing_requests_need_client_header_and_local_host():
    with TestClient(app) as c:
        assert c.delete("/api/history").status_code == 403
        assert c.post(f"/api/score/{KEYS[0]}", json={"features": {}}).status_code == 403
    with TestClient(app, base_url="http://evil.example") as c:
        assert c.get("/api/health").status_code == 400
