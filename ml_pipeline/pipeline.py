"""End-to-end inference used by the API and the evaluation scripts.

predict()  : track + road network -> features -> classifier -> HMM/Viterbi.
             Never reads the .route file.
evaluate() : server-side scoring of a prediction against the reference route.
"""
import json
import time

import joblib
import numpy as np

from . import config, io
from .features import FEATURES, prepare_trip
from .hmm import build_route, classifier_only, point_continuity, route_metrics, viterbi


class PathSnap:
    def __init__(self, artifact_dir=config.ARTIFACT_DIR):
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.cfg = json.loads((artifact_dir / "pipeline_config.json").read_text())
        assert self.cfg["features"] == FEATURES, "feature list changed since the model was exported"

    def predict(self, partition: str, trip_id: str) -> dict:
        timings = {}
        t0 = time.time()
        ctx = prepare_trip(partition, trip_id, **self.cfg["candidates"])
        timings["features_s"] = time.time() - t0

        t0 = time.time()
        ctx.frame["p"] = self.model.predict_proba(ctx.frame[FEATURES].to_numpy(np.float32))[:, 1]
        timings["classifier_s"] = time.time() - t0

        t0 = time.time()
        n = len(ctx.xy)
        clf_arcs = classifier_only(ctx.frame, n)
        hmm_arcs = viterbi(ctx.net, ctx.xy, ctx.t, ctx.frame, self.cfg["hmm"])
        route = build_route(ctx.net, hmm_arcs)
        timings["hmm_s"] = time.time() - t0
        return {"ctx": ctx, "classifier_arcs": clf_arcs, "hmm_arcs": hmm_arcs, "route": route, "timings": timings}


def evaluate(partition: str, trip_id: str, pred: dict) -> dict:
    """Compare a prediction with the server-side reference route."""
    from .labels import align_route
    ctx = pred["ctx"]
    ref_route = io.load_route(partition, trip_id)
    ref_arcs, ref_dist = align_route(ctx.xy, ref_route, ctx.net)
    reliable = ref_dist <= config.LABEL_MAX_DIST_M
    out = {
        "n_points": int(len(ref_arcs)),
        "n_scored_points": int(reliable.sum()),
        "hmm_point_accuracy": float((pred["hmm_arcs"] == ref_arcs)[reliable].mean()),
        "classifier_point_accuracy": float((pred["classifier_arcs"] == ref_arcs)[reliable].mean()),
        "hmm_continuity": point_continuity(ctx.net, pred["hmm_arcs"]),
        "classifier_continuity": point_continuity(ctx.net, pred["classifier_arcs"]),
    }
    out.update(route_metrics(ctx.net, pred["route"], ref_route))
    return out
