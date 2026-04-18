from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import _decorators as pandas_decorators
from sklearn.linear_model import LinearRegression


def _patch_pandas_deprecate_kwarg() -> None:
    signature = inspect.signature(pandas_decorators.deprecate_kwarg)
    if list(signature.parameters)[0] != "klass":
        return

    original = pandas_decorators.deprecate_kwarg

    def compat_deprecate_kwarg(
        old_arg_name: str,
        new_arg_name: str | None,
        mapping: Any = None,
        stacklevel: int = 2,
    ) -> Any:
        return original(
            FutureWarning,
            old_arg_name,
            new_arg_name,
            mapping=mapping,
            stacklevel=stacklevel,
        )

    pandas_decorators.deprecate_kwarg = compat_deprecate_kwarg


_patch_pandas_deprecate_kwarg()
from statsmodels.tsa.holtwinters import ExponentialSmoothing


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "backend" / "data" / "dataset.csv"
PRECOMPUTED_PATH = ROOT / "backend" / "data" / "precomputed.json"
FRONTEND_DATA_DIR = ROOT / "frontend" / "data"
TRAINING_DIR = ROOT / "backend" / "data" / "training"
EDA_SUMMARY_PATH = TRAINING_DIR / "eda_summary.json"
GPU_REGISTRY_PATH = ROOT / "backend" / "gpu_training" / "artifacts" / "model_registry.json"
INVALID_NOTES = {
    "duplicado",
    "missing",
    "missing_bloque",
    "negativo",
    "outlier_alto",
    "outlier_bajo",
}
IMPUTATION_METHODS = ("linear", "forward_fill", "seasonal_median")
MODEL_NAMES = ("seasonal_naive", "holt_winters", "seasonal_regression")
SEASONAL_PERIOD = 52
HOLDOUT_WEEKS = 12
MAX_HORIZON = 12
Z_90 = 1.6448536269514722


@dataclass
class ModelRun:
    name: str
    fitted: pd.Series
    forecast: pd.Series
    lower: pd.Series
    upper: pd.Series
    residuals: pd.Series


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1).dropna()
    if aligned.empty:
        return float("nan")
    return float(np.mean(np.abs(aligned.iloc[:, 0] - aligned.iloc[:, 1])))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    aligned = pd.concat([actual, predicted], axis=1).dropna()
    aligned = aligned.loc[aligned.iloc[:, 0] != 0]
    if aligned.empty:
        return float("nan")
    errors = np.abs((aligned.iloc[:, 0] - aligned.iloc[:, 1]) / aligned.iloc[:, 0]) * 100
    return float(np.mean(errors))


def residual_histogram(residuals: pd.Series, bins: int = 16) -> dict[str, Any]:
    values = residuals.dropna().to_numpy()
    if len(values) == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.1f}..{edges[i + 1]:.1f}" for i in range(len(edges) - 1)]
    return {"bins": labels, "counts": counts.astype(int).tolist()}


def load_dataset(csv_path: Path = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["fecha"])
    df["nota"] = df["nota"].fillna("missing")
    df = df.sort_values(["sku", "fecha"]).reset_index(drop=True)
    return df


def load_gpu_registry(registry_path: Path = GPU_REGISTRY_PATH) -> dict[str, Any] | None:
    if not registry_path.exists():
        return None
    return json.loads(registry_path.read_text(encoding="utf-8"))


def build_raw_series(df: pd.DataFrame, sku: str) -> pd.DataFrame:
    sku_df = df.loc[df["sku"] == sku, ["fecha", "sku", "producto", "unidades_vendidas", "nota"]].copy()
    if sku_df.empty:
        raise KeyError(f"SKU not found: {sku}")

    sku_df["is_invalid"] = sku_df["nota"].isin(INVALID_NOTES)
    sku_df["raw_value"] = sku_df["unidades_vendidas"]
    sku_df["base_clean_value"] = sku_df["raw_value"].where(~sku_df["is_invalid"])
    return sku_df


def prepare_clean_series(raw_df: pd.DataFrame, imputation_method: str) -> pd.DataFrame:
    prepared = raw_df.copy()
    cleaned = impute_series(prepared["base_clean_value"], prepared["fecha"], imputation_method)
    prepared["cleaned_value"] = cleaned.round(4)
    prepared["is_imputed"] = prepared["base_clean_value"].isna()
    prepared["was_anomaly"] = prepared["is_invalid"]
    prepared["iso_week"] = pd.to_datetime(prepared["fecha"]).dt.isocalendar().week.astype(int)
    prepared["month"] = pd.to_datetime(prepared["fecha"]).dt.month.astype(int)
    prepared["year"] = pd.to_datetime(prepared["fecha"]).dt.year.astype(int)
    prepared["time_index"] = np.arange(len(prepared), dtype=int)
    return prepared


def build_clean_dataset(
    df: pd.DataFrame,
    imputation_method: str = "seasonal_median",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sku in sorted(df["sku"].unique()):
        raw_df = build_raw_series(df, sku)
        clean_df = prepare_clean_series(raw_df, imputation_method)
        clean_df["imputation_method"] = imputation_method
        frames.append(clean_df)
    return pd.concat(frames, ignore_index=True)


def impute_series(base_series: pd.Series, dates: pd.Series, method: str) -> pd.Series:
    series = base_series.astype(float).copy()
    if method == "linear":
        return series.interpolate(method="linear", limit_direction="both").bfill().ffill()
    if method == "forward_fill":
        return series.ffill().bfill()
    if method == "seasonal_median":
        week_key = pd.to_datetime(dates).dt.isocalendar().week.astype(int)
        seasonal_map = pd.Series(series.values, index=week_key).groupby(level=0).median()
        filled = series.copy()
        for idx, value in filled.items():
            if pd.isna(value):
                filled.loc[idx] = seasonal_map.get(int(week_key.loc[idx]), np.nan)
        return filled.interpolate(method="linear", limit_direction="both").bfill().ffill()
    raise ValueError(f"Unsupported imputation method: {method}")


def fit_seasonal_naive(series: pd.Series, horizon: int) -> ModelRun:
    fitted = series.shift(SEASONAL_PERIOD)
    tail = series.iloc[-SEASONAL_PERIOD:]
    forecast_values = [float(tail.iloc[i % len(tail)]) for i in range(horizon)]
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    forecast = pd.Series(forecast_values, index=forecast_index, name="forecast")
    residuals = (series - fitted).dropna()
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0
    spread = pd.Series([Z_90 * sigma * math.sqrt(step + 1) for step in range(horizon)], index=forecast_index)
    return ModelRun(
        name="seasonal_naive",
        fitted=fitted,
        forecast=forecast,
        lower=forecast - spread,
        upper=forecast + spread,
        residuals=residuals,
    )


def fit_holt_winters(series: pd.Series, horizon: int) -> ModelRun:
    model = ExponentialSmoothing(
        series,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    )
    fitted_model = model.fit(optimized=True, use_brute=False)
    forecast = fitted_model.forecast(horizon)
    forecast.index = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    residuals = series - fitted_model.fittedvalues
    sigma = float(residuals.dropna().std(ddof=1)) if residuals.dropna().shape[0] > 1 else 0.0
    spread = pd.Series([Z_90 * sigma * math.sqrt(step + 1) for step in range(horizon)], index=forecast.index)
    return ModelRun(
        name="holt_winters",
        fitted=fitted_model.fittedvalues,
        forecast=forecast,
        lower=forecast - spread,
        upper=forecast + spread,
        residuals=residuals.dropna(),
    )


def fit_seasonal_regression(series: pd.Series, horizon: int) -> ModelRun:
    total_length = len(series)
    time_index = np.arange(total_length + horizon)
    trend = time_index.reshape(-1, 1)
    seasonal_features = []
    for harmonic in (1, 2, 3):
        seasonal_features.append(np.sin(2 * np.pi * harmonic * time_index / SEASONAL_PERIOD))
        seasonal_features.append(np.cos(2 * np.pi * harmonic * time_index / SEASONAL_PERIOD))
    features = np.column_stack([trend, *seasonal_features])
    model = LinearRegression()
    model.fit(features[:total_length], series.to_numpy())

    fitted_values = model.predict(features[:total_length])
    forecast_values = model.predict(features[total_length:])
    forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    fitted = pd.Series(fitted_values, index=series.index)
    forecast = pd.Series(forecast_values, index=forecast_index)
    residuals = series - fitted
    sigma = float(residuals.dropna().std(ddof=1)) if residuals.dropna().shape[0] > 1 else 0.0
    spread = pd.Series([Z_90 * sigma * math.sqrt(step + 1) for step in range(horizon)], index=forecast_index)
    return ModelRun(
        name="seasonal_regression",
        fitted=fitted,
        forecast=forecast,
        lower=forecast - spread,
        upper=forecast + spread,
        residuals=residuals.dropna(),
    )


def run_model(model_name: str, series: pd.Series, horizon: int) -> ModelRun:
    if model_name == "seasonal_naive":
        return fit_seasonal_naive(series, horizon)
    if model_name == "holt_winters":
        return fit_holt_winters(series, horizon)
    if model_name == "seasonal_regression":
        return fit_seasonal_regression(series, horizon)
    raise ValueError(f"Unsupported model: {model_name}")


def align_series_to_dates(series: pd.Series, dates: pd.Series) -> list[float | None]:
    aligned = series.reindex(pd.Index(dates)).tolist()
    normalized: list[float | None] = []
    for value in aligned:
        if pd.isna(value):
            normalized.append(None)
        else:
            normalized.append(round(float(value), 1))
    return normalized


def records_from_run(run: ModelRun, horizon: int) -> dict[str, Any]:
    forecast_rows = []
    for idx in run.forecast.index[:horizon]:
        forecast_rows.append(
            {
                "fecha": idx.strftime("%Y-%m-%d"),
                "valor_central": round(float(run.forecast.loc[idx]), 1),
                "ic_inferior": round(float(run.lower.loc[idx]), 1),
                "ic_superior": round(float(run.upper.loc[idx]), 1),
            }
        )
    return {
        "fitted": align_series_to_dates(run.fitted, run.fitted.index),
        "forecast": forecast_rows,
        "residuals": [round(float(value), 2) for value in run.residuals.dropna().tolist()],
        "residual_histogram": residual_histogram(run.residuals),
    }


def analyze_series(raw_df: pd.DataFrame, imputation_method: str, horizon: int = MAX_HORIZON) -> dict[str, Any]:
    prepared = prepare_clean_series(raw_df, imputation_method)
    dates = pd.date_range(prepared["fecha"].min(), periods=len(prepared), freq="W-MON")
    cleaned = prepared["cleaned_value"].astype(float).copy()
    cleaned.index = dates
    holdout = min(HOLDOUT_WEEKS, max(4, len(cleaned) // 8))
    train = cleaned.iloc[:-holdout]
    test = cleaned.iloc[-holdout:]

    comparison: list[dict[str, Any]] = []
    full_runs: dict[str, ModelRun] = {}
    best_model_name: str | None = None
    best_mape = float("inf")

    for model_name in MODEL_NAMES:
        try:
            validation_run = run_model(model_name, train, holdout)
            actual = test
            predicted = validation_run.forecast.reindex(actual.index)
            model_mae = mae(actual, predicted)
            model_mape = mape(actual, predicted)
            full_run = run_model(model_name, cleaned, horizon)
        except Exception as exc:
            comparison.append(
                {
                    "modelo": model_name,
                    "mae": None,
                    "mape": None,
                    "status": f"error: {exc}",
                }
            )
            continue

        comparison.append(
            {
                "modelo": model_name,
                "mae": round(model_mae, 2),
                "mape": round(model_mape, 2),
                "status": "ok",
            }
        )
        full_runs[model_name] = full_run
        if model_mape < best_mape:
            best_mape = model_mape
            best_model_name = model_name

    if best_model_name is None:
        raise RuntimeError("No model finished successfully.")

    best_run = full_runs[best_model_name]
    cleaned_values = [round(float(value), 1) for value in cleaned.tolist()]

    return {
        "imputation_method": imputation_method,
        "cleaned_values": cleaned_values,
        "cleaned_summary": {
            "media": round(float(np.mean(cleaned_values)), 2),
            "desviacion_estandar": round(float(np.std(cleaned_values, ddof=1)), 2),
        },
        "model_comparison": comparison,
        "best_model": best_model_name,
        "best_model_metrics": next(item for item in comparison if item["modelo"] == best_model_name),
        "best_run": records_from_run(best_run, horizon),
    }


def merge_gpu_analysis(
    analysis: dict[str, Any],
    gpu_entry: dict[str, Any] | None,
    horizon: int,
) -> dict[str, Any]:
    if not gpu_entry:
        return analysis

    merged = dict(analysis)
    comparison = [item for item in analysis["model_comparison"] if item["modelo"] != gpu_entry["best_model_metrics"]["modelo"]]
    comparison.append(gpu_entry["best_model_metrics"])
    merged["model_comparison"] = comparison

    gpu_best_run = gpu_entry.get("best_run")
    if gpu_best_run and gpu_entry["best_model_metrics"]["mape"] < analysis["best_model_metrics"]["mape"]:
        merged["best_model"] = gpu_entry["best_model_metrics"]["modelo"]
        merged["best_model_metrics"] = gpu_entry["best_model_metrics"]
        merged["best_run"] = {
            **gpu_best_run,
            "forecast": gpu_best_run["forecast"][:horizon],
        }

    return merged


def build_sku_response(
    sku: str,
    imputation_method: str = "seasonal_median",
    horizon: int = MAX_HORIZON,
    df: pd.DataFrame | None = None,
    gpu_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = df if df is not None else load_dataset()
    raw_df = build_raw_series(frame, sku)
    gpu_data = gpu_registry if gpu_registry is not None else load_gpu_registry()
    analysis = analyze_series(raw_df, imputation_method, horizon=horizon)
    gpu_entry = None
    if gpu_data is not None:
        gpu_entry = (
            gpu_data.get("imputations", {})
            .get(imputation_method, {})
            .get("skus", {})
            .get(sku)
        )
    analysis = merge_gpu_analysis(analysis, gpu_entry, horizon=horizon)

    raw_records = []
    for row in raw_df.itertuples(index=False):
        raw_records.append(
            {
                "fecha": row.fecha.strftime("%Y-%m-%d"),
                "valor": None if pd.isna(row.raw_value) else round(float(row.raw_value), 1),
                "nota": row.nota,
                "es_anomalia": bool(row.is_invalid),
            }
        )

    anomaly_log = [
        {
            "fecha": item["fecha"],
            "tipo": item["nota"],
            "valor_original": item["valor"],
        }
        for item in raw_records
        if item["nota"] != "ok"
    ]

    return {
        "sku": sku,
        "producto": str(raw_df["producto"].iloc[0]),
        "dates": [item["fecha"] for item in raw_records],
        "raw_series": raw_records,
        "anomaly_log": anomaly_log,
        "anomaly_summary": (
            pd.Series([entry["tipo"] for entry in anomaly_log]).value_counts().sort_index().to_dict()
            if anomaly_log
            else {}
        ),
        "analysis": analysis,
    }


def build_payload(df: pd.DataFrame | None = None, horizon: int = MAX_HORIZON) -> dict[str, Any]:
    frame = df if df is not None else load_dataset()
    gpu_registry = load_gpu_registry()
    payload: dict[str, Any] = {
        "meta": {
            "generated_from": "dataset.pdf",
            "source_csv": str(DATASET_PATH.relative_to(ROOT)),
            "weeks": int(frame["fecha"].nunique()),
            "sku_count": int(frame["sku"].nunique()),
            "date_range": {
                "inicio": frame["fecha"].min().strftime("%Y-%m-%d"),
                "fin": frame["fecha"].max().strftime("%Y-%m-%d"),
            },
            "imputation_methods": list(IMPUTATION_METHODS),
            "max_horizon": horizon,
        },
        "catalog": [],
        "series_by_sku": {},
    }

    for sku, sku_df in frame.groupby("sku"):
        raw_df = build_raw_series(frame, sku)
        product_name = str(raw_df["producto"].iloc[0])
        raw_records = []
        for row in raw_df.itertuples(index=False):
            raw_records.append(
                {
                    "fecha": row.fecha.strftime("%Y-%m-%d"),
                    "valor": None if pd.isna(row.raw_value) else round(float(row.raw_value), 1),
                    "nota": row.nota,
                    "es_anomalia": bool(row.is_invalid),
                }
            )

        anomaly_log = [
            {
                "fecha": item["fecha"],
                "tipo": item["nota"],
                "valor_original": item["valor"],
            }
            for item in raw_records
            if item["nota"] != "ok"
        ]

        analyses: dict[str, Any] = {}
        for method in IMPUTATION_METHODS:
            analysis = analyze_series(raw_df, method, horizon=horizon)
            gpu_entry = None
            if gpu_registry is not None:
                gpu_entry = (
                    gpu_registry.get("imputations", {})
                    .get(method, {})
                    .get("skus", {})
                    .get(sku)
                )
            analyses[method] = merge_gpu_analysis(analysis, gpu_entry, horizon=horizon)

        payload["catalog"].append({"sku": sku, "producto": product_name})
        payload["series_by_sku"][sku] = {
            "sku": sku,
            "producto": product_name,
            "dates": [item["fecha"] for item in raw_records],
            "raw_series": raw_records,
            "anomaly_log": anomaly_log,
            "anomaly_summary": (
                pd.Series([entry["tipo"] for entry in anomaly_log]).value_counts().sort_index().to_dict()
                if anomaly_log
                else {}
            ),
            "analyses": analyses,
        }

    payload["catalog"] = sorted(payload["catalog"], key=lambda item: item["sku"])
    return payload


def write_payload(payload: dict[str, Any], destination: Path = PRECOMPUTED_PATH) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_frontend_artifacts(payload: dict[str, Any], output_dir: Path = FRONTEND_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "catalog.json").write_text(
        json.dumps({"meta": payload["meta"], "catalog": payload["catalog"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    series_dir = output_dir / "series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for sku, data in payload["series_by_sku"].items():
        (series_dir / f"{sku}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if EDA_SUMMARY_PATH.exists():
        (output_dir / "eda-summary.json").write_text(
            EDA_SUMMARY_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def load_precomputed(precomputed_path: Path = PRECOMPUTED_PATH) -> dict[str, Any]:
    return json.loads(precomputed_path.read_text(encoding="utf-8"))
