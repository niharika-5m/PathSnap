"""PathSnap inference API (runs on Laptop 1).

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            model + config status
  GET  /trips             IDs of the evaluation trips (no labels)
  POST /predict           {"trip_id": "..."} -> predicted route + server-side scores

The reference route is read only inside pathsnap.pipeline.evaluate(), after the
prediction is finished. Only aggregate scores go back to the UI; the reference
geometry never leaves this machine.
"""
import logging
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import io
from pathsnap.hmm import build_route
from pathsnap.pipeline import PathSnap, evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap.api")

PARTITION = os.environ.get("PATHSNAP_EVAL_PARTITION", "evaluator_test")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine = PathSnap()
log.info("Loaded model '%s' (%s). Serving partition '%s' with %d trips.",
         engine.cfg["model_name"], engine.cfg["params_source"], PARTITION, len(io.list_trips(PARTITION)))


class PredictRequest(BaseModel):
    """Request schema for trip trajectory prediction and server-side evaluation."""
    trip_id: str


def _round(a, nd: int = 6):
    """Round coordinate arrays to fixed decimal precision (default 6 ~ 11cm)."""
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def _route_polylines(net, route):
    """Convert arc sequence into continuous [lat, lon] polylines.
    Splits into separate sub-polylines whenever adjacent arcs are topologically disconnected.
    """
    lines, cur = [], []
    for k, a in enumerate(route):
        s, d = int(net.src[a]), int(net.dst[a])
        if not cur or (k > 0 and int(net.dst[route[k - 1]]) != s):
            if cur:
                lines.append(cur)
            cur = [net.nodes_lonlat[s][::-1]]
        cur.append(net.nodes_lonlat[d][::-1])
    if cur:
        lines.append(cur)
    return [_round(np.array(line)) for line in lines]


def _arcs_geometry(net, arcs):
    """Extract distinct road segments as line coordinates [[lat1, lon1], [lat2, lon2]]."""
    arcs = np.asarray(sorted(set(int(a) for a in arcs)))
    if len(arcs) == 0:
        return []
    a = net.nodes_lonlat[net.src[arcs]][:, ::-1]
    b = net.nodes_lonlat[net.dst[arcs]][:, ::-1]
    return np.round(np.stack([a, b], axis=1), 6).tolist()


@app.get("/")
def root():
    """Service status landing endpoint for network diagnostic pings."""
    return {
        "service": "PathSnap API",
        "status": "running",
        "partition": PARTITION,
        "n_trips": len(io.list_trips(PARTITION)),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Health check returning loaded ML model type, HMM tuning parameters, and partition size."""
    return {"status": "ok", "model": engine.cfg["model_name"], "params_source": engine.cfg["params_source"],
            "partition": PARTITION, "n_trips": len(io.list_trips(PARTITION)), "hmm": engine.cfg["hmm"]}


@app.get("/trips")
def trips():
    """Enumerate available evaluation trip identifiers without exposing reference routes (zero leakage)."""
    ids = io.list_trips(PARTITION)
    log.info("GET /trips -> %d trips", len(ids))
    return {"partition": PARTITION, "trips": ids}


@app.post("/predict")
def predict(req: PredictRequest):
    """Execute end-to-end inference and zero-leakage server-side scoring for a single trip."""
    if req.trip_id not in io.list_trips(PARTITION):
        raise HTTPException(status_code=404, detail=f"unknown trip {req.trip_id}")
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)

    # 1. End-to-end inference (features -> classifier -> HMM Viterbi)
    t0 = time.time()
    pred = engine.predict(PARTITION, req.trip_id)
    predict_s = time.time() - t0
    ctx = pred["ctx"]

    # 2. Server-side zero-leakage evaluation against secret ground truth
    t1 = time.time()
    scores = evaluate(PARTITION, req.trip_id, pred)   # server-side only
    eval_s = time.time() - t1
    log.info("trip=%s points=%d candidates=%d | classifier acc=%.4f | HMM acc=%.4f | route F1=%.4f | "
             "predict %.2fs (features %.2fs, classifier %.2fs, hmm %.2fs) eval %.2fs",
             req.trip_id, scores["n_points"], len(ctx.frame), scores["classifier_point_accuracy"],
             scores["hmm_point_accuracy"], scores["route_f1"], predict_s, pred["timings"]["features_s"],
             pred["timings"]["classifier_s"], pred["timings"]["hmm_s"], eval_s)

    # 3. Build baseline classifier route and format GeoJSON response payload
    clf_route = build_route(ctx.net, pred["classifier_arcs"])
    return {
        "trip_id": req.trip_id,
        "model": engine.cfg["model_name"],
        "n_points": int(len(ctx.xy)),
        "gps": _round(np.c_[ctx.lat, ctx.lon]),
        "predicted_route": _route_polylines(ctx.net, pred["route"]),
        "classifier_route": _route_polylines(ctx.net, clf_route),
        "nearby_roads": _arcs_geometry(ctx.net, ctx.frame["arc"].to_numpy()),
        "n_route_arcs": len(pred["route"]),
        "metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in scores.items()},
        "timings": {**{k: round(v, 3) for k, v in pred["timings"].items()},
                    "predict_total_s": round(predict_s, 3), "evaluation_s": round(eval_s, 3)},
    }
