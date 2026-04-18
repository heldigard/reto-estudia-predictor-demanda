from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .forecasting import (
    HOLDOUT_WEEKS,
    IMPUTATION_METHODS,
    MAX_HORIZON,
    TRAINING_DIR,
    analyze_series,
    build_clean_dataset,
    build_raw_series,
    load_dataset,
)


EDA_SUMMARY_PATH = TRAINING_DIR / "eda_summary.json"
EDA_REPORT_PATH = TRAINING_DIR / "eda_report.md"
CLEAN_DATASET_PATH = TRAINING_DIR / "demand_history_clean.csv"
TRAIN_READY_DATASET_PATH = TRAINING_DIR / "demand_training_features.csv"
MODEL_REGISTRY_PATH = TRAINING_DIR / "model_registry.json"


def longest_consecutive_streak(mask: pd.Series) -> int:
    max_streak = 0
    current = 0
    for value in mask.astype(bool).tolist():
        if value:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return int(max_streak)


def build_eda_summary(df: pd.DataFrame, clean_df: pd.DataFrame) -> dict[str, Any]:
    raw_note_counts = df["nota"].value_counts().sort_index().to_dict()
    invalid_mask = df["nota"] != "ok"
    weekly_invalid_rate = (
        df.assign(is_invalid=invalid_mask)
        .groupby("fecha", as_index=False)["is_invalid"]
        .mean()
        .rename(columns={"is_invalid": "invalid_rate"})
    )

    sku_profiles = []
    for sku, sku_df in df.groupby("sku"):
        clean_sku = clean_df.loc[clean_df["sku"] == sku]
        invalid = sku_df["nota"] != "ok"
        sku_profiles.append(
            {
                "sku": sku,
                "producto": str(sku_df["producto"].iloc[0]),
                "rows": int(len(sku_df)),
                "invalid_rows": int(invalid.sum()),
                "invalid_rate_pct": round(float(invalid.mean() * 100), 2),
                "missing_streak_max": longest_consecutive_streak(sku_df["nota"].isin({"missing", "missing_bloque"})),
                "clean_mean": round(float(clean_sku["cleaned_value"].mean()), 2),
                "clean_std": round(float(clean_sku["cleaned_value"].std(ddof=1)), 2),
                "clean_min": round(float(clean_sku["cleaned_value"].min()), 2),
                "clean_max": round(float(clean_sku["cleaned_value"].max()), 2),
            }
        )

    top_invalid_skus = sorted(sku_profiles, key=lambda item: item["invalid_rate_pct"], reverse=True)[:5]

    return {
        "dataset": {
            "rows": int(len(df)),
            "weeks": int(df["fecha"].nunique()),
            "skus": int(df["sku"].nunique()),
            "date_start": df["fecha"].min().strftime("%Y-%m-%d"),
            "date_end": df["fecha"].max().strftime("%Y-%m-%d"),
            "holdout_weeks": HOLDOUT_WEEKS,
        },
        "raw_note_counts": raw_note_counts,
        "invalid_rate_pct_global": round(float((invalid_mask.mean()) * 100), 2),
        "weekly_invalid_rate_summary": {
            "mean_pct": round(float(weekly_invalid_rate["invalid_rate"].mean() * 100), 2),
            "max_pct": round(float(weekly_invalid_rate["invalid_rate"].max() * 100), 2),
            "worst_week": weekly_invalid_rate.sort_values("invalid_rate", ascending=False).iloc[0]["fecha"].strftime("%Y-%m-%d"),
        },
        "top_invalid_skus": top_invalid_skus,
        "sku_profiles": sku_profiles,
    }


def eda_summary_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# EDA Summary",
        "",
        "## Dataset",
        "",
        f"- filas: {dataset['rows']}",
        f"- semanas: {dataset['weeks']}",
        f"- sku: {dataset['skus']}",
        f"- rango: {dataset['date_start']} -> {dataset['date_end']}",
        f"- holdout configurado: {dataset['holdout_weeks']} semanas",
        "",
        "## Calidad del dato",
        "",
        f"- porcentaje global inválido: {summary['invalid_rate_pct_global']}%",
        f"- peor semana por tasa inválida: {summary['weekly_invalid_rate_summary']['worst_week']}",
        f"- peor tasa semanal inválida: {summary['weekly_invalid_rate_summary']['max_pct']}%",
        "",
        "## Conteo por nota",
        "",
    ]

    for note, count in summary["raw_note_counts"].items():
        lines.append(f"- {note}: {count}")

    lines.extend(
        [
            "",
            "## SKU con mayor fricción de limpieza",
            "",
        ]
    )

    for item in summary["top_invalid_skus"]:
        lines.append(
            f"- {item['sku']} ({item['producto']}): {item['invalid_rate_pct']}% inválido, "
            f"racha missing máxima {item['missing_streak_max']} semanas, media limpia {item['clean_mean']}"
        )

    return "\n".join(lines) + "\n"


def build_training_registry(df: pd.DataFrame, horizon: int = MAX_HORIZON) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "meta": {
            "horizon": horizon,
            "candidate_models": ["seasonal_naive", "holt_winters", "seasonal_regression"],
            "candidate_imputations": list(IMPUTATION_METHODS),
        },
        "models_by_sku": {},
    }

    for sku in sorted(df["sku"].unique()):
        raw_df = build_raw_series(df, sku)
        per_method = {}
        best_entry: dict[str, Any] | None = None

        for method in IMPUTATION_METHODS:
            analysis = analyze_series(raw_df, method, horizon=horizon)
            entry = {
                "imputation_method": method,
                "best_model": analysis["best_model"],
                "best_model_metrics": analysis["best_model_metrics"],
                "model_comparison": analysis["model_comparison"],
                "cleaned_summary": analysis["cleaned_summary"],
                "forecast_preview": analysis["best_run"]["forecast"][:3],
            }
            per_method[method] = entry

            current_mape = analysis["best_model_metrics"]["mape"]
            if best_entry is None or current_mape < best_entry["best_model_metrics"]["mape"]:
                best_entry = entry

        registry["models_by_sku"][sku] = {
            "sku": sku,
            "producto": str(raw_df["producto"].iloc[0]),
            "best_training_recipe": best_entry,
            "recipes_by_imputation": per_method,
        }

    return registry


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_training_pipeline() -> dict[str, Path]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    clean_df = build_clean_dataset(df, imputation_method="seasonal_median")
    train_ready_df = clean_df[
        [
            "fecha",
            "sku",
            "producto",
            "nota",
            "raw_value",
            "base_clean_value",
            "cleaned_value",
            "is_imputed",
            "was_anomaly",
            "iso_week",
            "month",
            "year",
            "time_index",
        ]
    ].copy()

    summary = build_eda_summary(df, clean_df)
    registry = build_training_registry(df, horizon=MAX_HORIZON)

    clean_df.to_csv(CLEAN_DATASET_PATH, index=False)
    train_ready_df.to_csv(TRAIN_READY_DATASET_PATH, index=False)
    write_json(EDA_SUMMARY_PATH, summary)
    EDA_REPORT_PATH.write_text(eda_summary_markdown(summary), encoding="utf-8")
    write_json(MODEL_REGISTRY_PATH, registry)

    return {
        "clean_dataset": CLEAN_DATASET_PATH,
        "train_ready_dataset": TRAIN_READY_DATASET_PATH,
        "eda_summary": EDA_SUMMARY_PATH,
        "eda_report": EDA_REPORT_PATH,
        "model_registry": MODEL_REGISTRY_PATH,
    }
