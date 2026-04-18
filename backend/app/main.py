from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .forecasting import MAX_HORIZON, PRECOMPUTED_PATH, build_payload, load_precomputed, write_payload


ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title="Predictor de Demanda API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_payload() -> dict[str, Any]:
    if PRECOMPUTED_PATH.exists():
        return load_precomputed()
    payload = build_payload()
    write_payload(payload)
    return payload


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    payload = get_payload()
    return {"meta": payload["meta"], "catalog": payload["catalog"]}


@app.get("/api/sku/{sku}")
def sku_details(
    sku: str,
    imputation: str = Query("seasonal_median"),
    horizon: int = Query(MAX_HORIZON, ge=1, le=MAX_HORIZON),
) -> dict[str, Any]:
    payload = get_payload()
    sku_data = payload["series_by_sku"].get(sku)
    if sku_data is None:
        raise HTTPException(status_code=404, detail=f"SKU not found: {sku}")

    analysis = sku_data["analyses"].get(imputation)
    if analysis is None:
        raise HTTPException(status_code=400, detail=f"Unsupported imputation method: {imputation}")

    response = {
        "sku": sku_data["sku"],
        "producto": sku_data["producto"],
        "dates": sku_data["dates"],
        "raw_series": sku_data["raw_series"],
        "anomaly_log": sku_data["anomaly_log"],
        "anomaly_summary": sku_data["anomaly_summary"],
        "analysis": {
            **analysis,
            "best_run": {
                **analysis["best_run"],
                "forecast": analysis["best_run"]["forecast"][:horizon],
            },
        },
    }
    return response
