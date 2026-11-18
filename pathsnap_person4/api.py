"""Laptop-1 FastAPI server. No hardcoded routes, predictions or dummy responses."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import re
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .metrics import evaluate_predictions, load_point_labels
from .pipeline import NotConfigured, predictor_from_environment

logger = logging.getLogger("pathsnap.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
ID_PATTERN = re.compile(r"^\d{8}$")


def data_root() -> Path:
    raw = os.getenv("PATHSNAP_DATA_ROOT")
    if not raw:
        raise NotConfigured("Set PATHSNAP_DATA_ROOT to the folder containing the trip subfolders")
    result = Path(raw)
    if not result.is_dir():
        raise NotConfigured(f"Trip directory not found: {result}")
    return result


def known_ids() -> list[str]:
    root = data_root()
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and ID_PATTERN.fullmatch(d.name)
        and all((d / f"{d.name}.{ext}").is_file() for ext in ("track", "nodes", "arcs"))
    )


@lru_cache(maxsize=1)
def get_predictor():
    return predictor_from_environment()


def _write_request_log(info: dict) -> None:
    location = os.getenv("PATHSNAP_REQUEST_LOG", "logs/requests.jsonl")
    log_file = Path(location)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(info, separators=(",", ":")) + "\n")


def create_app() -> FastAPI:
    application = FastAPI(title="PathSnap map-matching API", version="1.0.0")
    # The UI is hosted on another laptop on the same Wi-Fi / hotspot.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "*").split(",")],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @application.get("/health")
    def health():
        try:
            ids = known_ids()
            get_predictor()
        except (NotConfigured, ValueError) as exc:
            return {"status": "not_ready", "reason": str(exc)}
        return {"status": "ready", "available_trips": len(ids)}

    @application.get("/trips")
    def trips():
        try:
            ids = known_ids()
        except NotConfigured as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"trip_ids": ids, "count": len(ids)}

    @application.get("/predict/{trip_id}")
    def predict(trip_id: str):
        started = time.perf_counter()
        status, error = "ok", None
        try:
            if not ID_PATTERN.fullmatch(trip_id):
                raise HTTPException(status_code=400, detail="trip_id must be exactly 8 digits")
            if trip_id not in known_ids():
                raise HTTPException(status_code=404, detail="Unknown trip ID")
            predictor = get_predictor()
            output = predictor.predict(data_root() / trip_id)
            # This read occurs AFTER predictions and is separate from model and UI.
            # File is optional on deployment; .route is NEVER treated as point labels.
            reference_root = os.getenv("PATHSNAP_REFERENCE_LABELS_DIR")
            labels = (
                load_point_labels(Path(reference_root) / f"{trip_id}.csv", output["observation_count"])
                if reference_root else None
            )
            output["metrics"] = evaluate_predictions(output, labels)
            return output
        except NotConfigured as exc:
            status, error = "not_configured", str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException as exc:
            status, error = f"http_{exc.status_code}", str(exc.detail)
            raise
        except (ValueError, FileNotFoundError) as exc:
            status, error = "bad_data", str(exc)
            logger.exception("Bad data / artifact configuration for trip %s", trip_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            status, error = "internal_error", "internal error"
            logger.exception("Inference failed for %s", trip_id)
            raise HTTPException(status_code=500, detail="Inference failed; consult Laptop 1 logs")
        finally:
            duration = round(time.perf_counter() - started, 4)
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "endpoint": "/predict/{trip_id}",
                "trip_id": trip_id,
                "status": status,
                "duration_seconds": duration,
                "error": error,
            }
            logger.info("PREDICT trip=%s status=%s elapsed=%.4fs", trip_id, status, duration)
            try:
                _write_request_log(event)
            except OSError:
                logger.exception("Unable to write request audit log")

    return application


app = create_app()
