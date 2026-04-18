from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from backend.app.forecasting import (
    IMPUTATION_METHODS,
    MAX_HORIZON,
    build_clean_dataset,
    load_dataset,
    mae,
    mape,
    residual_histogram,
    run_model,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "backend" / "gpu_training" / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CONFIG_PATH = ARTIFACTS_DIR / "config.json"
METRICS_SUMMARY_PATH = ARTIFACTS_DIR / "metrics_summary.json"
EXPERIMENT_REPORT_PATH = ARTIFACTS_DIR / "experiment_report.md"
MODEL_REGISTRY_PATH = ARTIFACTS_DIR / "model_registry.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

MODEL_NAME = "catboost_gpu_global"
SEASONAL_PERIOD = 52
TRAIN_WEEKS = 520 - 24
VALIDATION_WEEKS = 12
TEST_WEEKS = 12
CONFIDENCE_Z = 1.6448536269514722
HYPERPARAM_GRID = [
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5, "iterations": 500},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 8, "iterations": 700},
    {"depth": 10, "learning_rate": 0.03, "l2_leaf_reg": 10, "iterations": 1000},
]
FEATURE_COLUMNS = [
    "sku",
    "producto",
    "year",
    "month",
    "iso_week",
    "time_index",
    "is_imputed",
    "was_anomaly",
    "horizon",
    "sin_week",
    "cos_week",
    "sin_month",
    "cos_month",
    "target_iso_week",
    "target_month",
    "target_sin_week",
    "target_cos_week",
    "target_sin_month",
    "target_cos_month",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_12",
    "lag_26",
    "lag_52",
    "roll_mean_4",
    "roll_std_4",
    "roll_min_4",
    "roll_max_4",
    "roll_mean_8",
    "roll_std_8",
    "roll_min_8",
    "roll_max_8",
    "roll_mean_12",
    "roll_std_12",
    "roll_min_12",
    "roll_max_12",
    "roll_mean_26",
    "roll_std_26",
    "roll_min_26",
    "roll_max_26",
    "roll_mean_52",
    "roll_std_52",
    "roll_min_52",
    "roll_max_52",
    "delta_vs_lag_1",
    "pct_vs_lag_4",
    "pct_vs_lag_52",
]
CATEGORICAL_COLUMNS = ["sku", "producto"]


def rmse(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual_array - predicted_array) ** 2)))


def smape(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    valid = denominator != 0
    if not valid.any():
        return 0.0
    return float(np.mean((2.0 * np.abs(predicted_array[valid] - actual_array[valid]) / denominator[valid])) * 100)


def metric_payload(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray, model_name: str) -> dict[str, Any]:
    actual_series = pd.Series(actual)
    predicted_series = pd.Series(predicted)
    return {
        "modelo": model_name,
        "mae": round(mae(actual_series, predicted_series), 2),
        "mape": round(mape(actual_series, predicted_series), 2),
        "rmse": round(rmse(actual_series, predicted_series), 2),
        "smape": round(smape(actual_series, predicted_series), 2),
        "status": "ok",
    }


def split_dates(clean_df: pd.DataFrame) -> dict[str, pd.Timestamp]:
    all_dates = np.sort(clean_df["fecha"].unique())
    return {
        "train_end": pd.Timestamp(all_dates[TRAIN_WEEKS - 1]),
        "validation_end": pd.Timestamp(all_dates[TRAIN_WEEKS + VALIDATION_WEEKS - 1]),
        "test_end": pd.Timestamp(all_dates[TRAIN_WEEKS + VALIDATION_WEEKS + TEST_WEEKS - 1]),
    }


def build_feature_frame(clean_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, sku_df in clean_df.groupby("sku"):
        sku_df = sku_df.sort_values("fecha").copy()
        values = sku_df["cleaned_value"].astype(float)
        iso_week = sku_df["iso_week"].astype(float)
        month = sku_df["month"].astype(float)

        sku_df["sin_week"] = np.sin(2 * np.pi * iso_week / SEASONAL_PERIOD)
        sku_df["cos_week"] = np.cos(2 * np.pi * iso_week / SEASONAL_PERIOD)
        sku_df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
        sku_df["cos_month"] = np.cos(2 * np.pi * month / 12.0)

        for lag in (1, 2, 4, 8, 12, 26, 52):
            sku_df[f"lag_{lag}"] = values.shift(lag)

        shifted = values.shift(1)
        for window in (4, 8, 12, 26, 52):
            sku_df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            sku_df[f"roll_std_{window}"] = shifted.rolling(window).std()
            sku_df[f"roll_min_{window}"] = shifted.rolling(window).min()
            sku_df[f"roll_max_{window}"] = shifted.rolling(window).max()

        sku_df["delta_vs_lag_1"] = values.shift(1) - values.shift(2)
        sku_df["pct_vs_lag_4"] = (values.shift(1) - values.shift(4)) / values.shift(4).replace(0, np.nan)
        sku_df["pct_vs_lag_52"] = (values.shift(1) - values.shift(52)) / values.shift(52).replace(0, np.nan)
        frames.append(sku_df)

    return pd.concat(frames, ignore_index=True)


def build_pooled_supervised_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for horizon in range(1, MAX_HORIZON + 1):
        horizon_frames: list[pd.DataFrame] = []
        for _, sku_df in feature_df.groupby("sku"):
            sku_df = sku_df.sort_values("fecha").copy()
            sku_df["target"] = sku_df["cleaned_value"].shift(-horizon)
            sku_df["target_fecha"] = sku_df["fecha"].shift(-horizon)
            sku_df["horizon"] = horizon
            sku_df["target_iso_week"] = sku_df["target_fecha"].dt.isocalendar().week.astype("float64")
            sku_df["target_month"] = sku_df["target_fecha"].dt.month.astype("float64")
            sku_df["target_sin_week"] = np.sin(2 * np.pi * sku_df["target_iso_week"] / SEASONAL_PERIOD)
            sku_df["target_cos_week"] = np.cos(2 * np.pi * sku_df["target_iso_week"] / SEASONAL_PERIOD)
            sku_df["target_sin_month"] = np.sin(2 * np.pi * sku_df["target_month"] / 12.0)
            sku_df["target_cos_month"] = np.cos(2 * np.pi * sku_df["target_month"] / 12.0)
            horizon_frames.append(sku_df)
        frames.append(pd.concat(horizon_frames, ignore_index=True))

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.dropna(subset=["target"])
    return pooled.dropna(subset=FEATURE_COLUMNS)


def build_future_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, last_row in feature_df.sort_values("fecha").groupby("sku").tail(1).iterrows():
        last_date = pd.Timestamp(last_row["fecha"])
        for horizon in range(1, MAX_HORIZON + 1):
            target_date = last_date + pd.Timedelta(weeks=horizon)
            row = last_row.to_dict()
            row["horizon"] = horizon
            row["target_fecha"] = target_date
            row["target_iso_week"] = float(target_date.isocalendar().week)
            row["target_month"] = float(target_date.month)
            row["target_sin_week"] = float(np.sin(2 * np.pi * row["target_iso_week"] / SEASONAL_PERIOD))
            row["target_cos_week"] = float(np.cos(2 * np.pi * row["target_iso_week"] / SEASONAL_PERIOD))
            row["target_sin_month"] = float(np.sin(2 * np.pi * row["target_month"] / 12.0))
            row["target_cos_month"] = float(np.cos(2 * np.pi * row["target_month"] / 12.0))
            rows.append(row)
    return pd.DataFrame(rows).dropna(subset=FEATURE_COLUMNS)


def classic_metrics_for_sku(clean_df: pd.DataFrame, split_info: dict[str, pd.Timestamp], sku: str) -> list[dict[str, Any]]:
    sku_df = clean_df.loc[clean_df["sku"] == sku].sort_values("fecha").copy()
    dates = pd.date_range(sku_df["fecha"].min(), periods=len(sku_df), freq="W-MON")
    series = pd.Series(sku_df["cleaned_value"].astype(float).tolist(), index=dates)
    trainval = series.loc[: split_info["validation_end"]]
    test = series.loc[(series.index > split_info["validation_end"]) & (series.index <= split_info["test_end"])]

    metrics = []
    for model_name in ("seasonal_naive", "holt_winters", "seasonal_regression"):
        run = run_model(model_name, trainval, len(test))
        predicted = run.forecast.reindex(test.index)
        metrics.append(metric_payload(test, predicted, model_name))
    return metrics


def global_classic_metrics(clean_df: pd.DataFrame, split_info: dict[str, pd.Timestamp]) -> dict[str, dict[str, Any]]:
    aggregate = {name: {"actual": [], "pred": []} for name in ("seasonal_naive", "holt_winters", "seasonal_regression")}
    for sku in sorted(clean_df["sku"].unique()):
        sku_df = clean_df.loc[clean_df["sku"] == sku].sort_values("fecha").copy()
        dates = pd.date_range(sku_df["fecha"].min(), periods=len(sku_df), freq="W-MON")
        series = pd.Series(sku_df["cleaned_value"].astype(float).tolist(), index=dates)
        trainval = series.loc[: split_info["validation_end"]]
        test = series.loc[(series.index > split_info["validation_end"]) & (series.index <= split_info["test_end"])]
        for model_name in aggregate:
            run = run_model(model_name, trainval, len(test))
            predicted = run.forecast.reindex(test.index)
            aggregate[model_name]["actual"].extend(test.tolist())
            aggregate[model_name]["pred"].extend(predicted.tolist())

    return {
        model_name: metric_payload(values["actual"], values["pred"], model_name)
        for model_name, values in aggregate.items()
    }


def fit_catboost(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trials: list[dict[str, Any]] = []
    best_score = float("inf")
    best_params: dict[str, Any] | None = None

    for params in HYPERPARAM_GRID:
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="MAPE",
            task_type="GPU",
            devices="0",
            random_seed=42,
            verbose=False,
            **params,
        )
        model.fit(train_df[FEATURE_COLUMNS], train_df["target"], cat_features=CATEGORICAL_COLUMNS)
        validation_pred = model.predict(validation_df[FEATURE_COLUMNS])
        metrics = metric_payload(validation_df["target"], validation_pred, MODEL_NAME)
        metrics["params"] = params
        trials.append(metrics)
        if metrics["mape"] < best_score:
            best_score = metrics["mape"]
            best_params = params

    assert best_params is not None
    return best_params, trials


def build_gpu_run(
    feature_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    model: CatBoostRegressor,
    split_info: dict[str, pd.Timestamp],
    sku: str,
    interval_sigmas: dict[int, float],
) -> dict[str, Any]:
    sku_history = feature_df.loc[feature_df["sku"] == sku].sort_values("fecha").copy()
    one_step_df = pooled_df.loc[(pooled_df["sku"] == sku) & (pooled_df["horizon"] == 1)].copy()
    one_step_pred = model.predict(one_step_df[FEATURE_COLUMNS])

    fitted_lookup = {
        pd.Timestamp(target_date): float(prediction)
        for target_date, prediction in zip(one_step_df["target_fecha"], one_step_pred)
    }
    fitted = []
    residual_values: list[float] = []
    for date in sku_history["fecha"]:
        prediction = fitted_lookup.get(pd.Timestamp(date))
        fitted.append(None if prediction is None else round(prediction, 1))

    for target_date, actual, prediction in zip(one_step_df["target_fecha"], one_step_df["target"], one_step_pred):
        timestamp = pd.Timestamp(target_date)
        if split_info["validation_end"] < timestamp <= split_info["test_end"]:
            residual_values.append(float(actual - prediction))

    future_df = build_future_rows(feature_df)
    sku_future = future_df.loc[future_df["sku"] == sku].sort_values("horizon").copy()
    future_pred = model.predict(sku_future[FEATURE_COLUMNS])
    forecast = []
    for row, prediction in zip(sku_future.itertuples(index=False), future_pred):
        sigma = interval_sigmas.get(int(row.horizon), 0.0)
        spread = CONFIDENCE_Z * sigma
        forecast.append(
            {
                "fecha": pd.Timestamp(row.target_fecha).strftime("%Y-%m-%d"),
                "valor_central": round(float(prediction), 1),
                "ic_inferior": round(float(prediction - spread), 1),
                "ic_superior": round(float(prediction + spread), 1),
            }
        )

    return {
        "fitted": fitted,
        "forecast": forecast,
        "residuals": [round(value, 2) for value in residual_values],
        "residual_histogram": residual_histogram(pd.Series(residual_values)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def report_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# GPU Training Report",
        "",
        "## 1. Resumen ejecutivo",
        "",
        summary["executive_summary"],
        "",
        "## 2. Decisiones de limpieza y features",
        "",
        "- Se construyó un modelo CatBoost GPU multiserie con `sku` y `producto` como categóricas.",
        "- Las features incluyen lags 1/2/4/8/12/26/52, ventanas rolling 4/8/12/26/52, estacionalidad cíclica y cambios porcentuales.",
        "- Se respetó un split temporal estricto: train hasta `t-24`, validación siguiente 12 semanas, test últimas 12 semanas.",
        "- Se evaluaron las imputaciones `linear`, `forward_fill` y `seasonal_median` para no sesgar el modelo GPU a una sola receta.",
        "",
        "## 3. Tabla comparativa de modelos",
        "",
    ]
    for method, method_summary in summary["imputations"].items():
        lines.append(f"### {method}")
        lines.append("")
        for model_name, metrics in method_summary["global_metrics"].items():
            lines.append(
                f"- {model_name}: MAPE {metrics['mape']} | MAE {metrics['mae']} | RMSE {metrics['rmse']} | sMAPE {metrics['smape']}"
            )
        lines.append("")
    lines.extend(
        [
            "## 4. Mejor modelo final y por qué",
            "",
            summary["final_recommendation"],
            "",
            "## 5. Riesgos y límites",
            "",
        ]
    )
    for risk in summary["risks"]:
        lines.append(f"- {risk}")
    lines.extend(
        [
            "",
            "## 6. Rutas exactas de archivos generados",
            "",
        ]
    )
    for label, path in summary["artifacts"].items():
        lines.append(f"- {label}: `{path}`")
    return "\n".join(lines) + "\n"


def run_gpu_training() -> dict[str, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    audit = {
        "rows": int(len(dataset)),
        "sku_count": int(dataset["sku"].nunique()),
        "week_count": int(dataset["fecha"].nunique()),
        "invalid_rate_pct": round(float((dataset["nota"] != "ok").mean() * 100), 2),
    }

    registry: dict[str, Any] = {
        "meta": {
            "model_name": MODEL_NAME,
            "audit": audit,
        },
        "imputations": {},
    }
    prediction_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "executive_summary": "",
        "final_recommendation": "",
        "risks": [
            "CatBoost GPU no domina globalmente a Holt-Winters o regresión estacional en este dataset.",
            "Las bandas del modelo GPU usan desviación estándar de residuales de validación por horizonte, no cuantiles dedicados.",
            "El modelo deep se omite por relación costo/beneficio: solo 20 series y 520 semanas por SKU.",
        ],
        "imputations": {},
        "artifacts": {},
    }

    gpu_wins = 0

    for method in IMPUTATION_METHODS:
        clean_df = build_clean_dataset(dataset, imputation_method=method)
        feature_df = build_feature_frame(clean_df)
        pooled_df = build_pooled_supervised_frame(feature_df)
        split_info = split_dates(clean_df)

        train_df = pooled_df.loc[pooled_df["target_fecha"] <= split_info["train_end"]]
        validation_df = pooled_df.loc[
            (pooled_df["target_fecha"] > split_info["train_end"]) &
            (pooled_df["target_fecha"] <= split_info["validation_end"])
        ]
        trainval_df = pooled_df.loc[pooled_df["target_fecha"] <= split_info["validation_end"]]
        test_df = pooled_df.loc[
            (pooled_df["target_fecha"] > split_info["validation_end"]) &
            (pooled_df["target_fecha"] <= split_info["test_end"])
        ].copy()

        best_params, trials = fit_catboost(train_df, validation_df)
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="MAPE",
            task_type="GPU",
            devices="0",
            random_seed=42,
            verbose=False,
            **best_params,
        )
        model.fit(trainval_df[FEATURE_COLUMNS], trainval_df["target"], cat_features=CATEGORICAL_COLUMNS)
        model_path = MODELS_DIR / f"{MODEL_NAME}_{method}.cbm"
        model.save_model(model_path)

        validation_pred = model.predict(validation_df[FEATURE_COLUMNS])
        validation_scored = validation_df.copy()
        validation_scored["prediction"] = validation_pred
        interval_sigmas = (
            validation_scored.assign(residual=validation_scored["target"] - validation_scored["prediction"])
            .groupby("horizon")["residual"]
            .std(ddof=1)
            .fillna(0.0)
            .to_dict()
        )

        test_pred = model.predict(test_df[FEATURE_COLUMNS])
        test_df["prediction"] = test_pred
        gpu_global_metrics = metric_payload(test_df["target"], test_df["prediction"], MODEL_NAME)

        classic_global = global_classic_metrics(clean_df, split_info)
        summary["imputations"][method] = {
            "selected_params": best_params,
            "global_metrics": {**classic_global, MODEL_NAME: gpu_global_metrics},
            "hyperparameter_trials": trials,
        }

        registry["imputations"][method] = {
            "model_path": str(model_path.relative_to(ROOT)),
            "selected_params": best_params,
            "global_metrics": {**classic_global, MODEL_NAME: gpu_global_metrics},
            "skus": {},
        }

        future_df = build_future_rows(feature_df)
        future_pred = model.predict(future_df[FEATURE_COLUMNS])
        future_df = future_df.copy()
        future_df["prediction"] = future_pred

        for sku in sorted(clean_df["sku"].unique()):
            sku_test = test_df.loc[test_df["sku"] == sku].copy()
            sku_gpu_metrics = metric_payload(sku_test["target"], sku_test["prediction"], MODEL_NAME)
            classic_metrics = classic_metrics_for_sku(clean_df, split_info, sku)
            best_classic = min(classic_metrics, key=lambda item: item["mape"])

            best_run = None
            if sku_gpu_metrics["mape"] < best_classic["mape"]:
                gpu_wins += 1
                best_run = build_gpu_run(
                    feature_df=feature_df,
                    pooled_df=pooled_df,
                    model=model,
                    split_info=split_info,
                    sku=sku,
                    interval_sigmas={int(key): float(value) for key, value in interval_sigmas.items()},
                )

            registry["imputations"][method]["skus"][sku] = {
                "best_model_metrics": sku_gpu_metrics,
                "best_run": best_run,
            }

            producto = str(clean_df.loc[clean_df["sku"] == sku, "producto"].iloc[0])
            for row in sku_test.itertuples(index=False):
                prediction_rows.append(
                    {
                        "phase": "test",
                        "imputation_method": method,
                        "sku": row.sku,
                        "producto": row.producto,
                        "fecha": pd.Timestamp(row.target_fecha).strftime("%Y-%m-%d"),
                        "horizon": int(row.horizon),
                        "actual": round(float(row.target), 2),
                        "prediction": round(float(row.prediction), 2),
                        "lower": None,
                        "upper": None,
                    }
                )

            sku_future = future_df.loc[future_df["sku"] == sku].sort_values("horizon")
            for row in sku_future.itertuples(index=False):
                sigma = float(interval_sigmas.get(int(row.horizon), 0.0))
                spread = CONFIDENCE_Z * sigma
                prediction_rows.append(
                    {
                        "phase": "forecast",
                        "imputation_method": method,
                        "sku": sku,
                        "producto": producto,
                        "fecha": pd.Timestamp(row.target_fecha).strftime("%Y-%m-%d"),
                        "horizon": int(row.horizon),
                        "actual": None,
                        "prediction": round(float(row.prediction), 2),
                        "lower": round(float(row.prediction - spread), 2),
                        "upper": round(float(row.prediction + spread), 2),
                    }
                )

    linear_metrics = summary["imputations"]["linear"]["global_metrics"]
    best_linear_name, best_linear_metrics = min(
        linear_metrics.items(),
        key=lambda item: item[1]["mape"],
    )
    summary["executive_summary"] = (
        "Se entrenó CatBoost GPU multiserie con validación temporal estricta. "
        f"En la imputación `linear`, el mejor baseline global fue `{best_linear_name}` con MAPE {best_linear_metrics['mape']}, "
        f"mientras `{MODEL_NAME}` logró MAPE {linear_metrics[MODEL_NAME]['mape']}. "
        f"Aun así, el modelo GPU ganó en {gpu_wins} combinaciones SKU/imputación y queda integrado como candidato opcional."
    )
    summary["final_recommendation"] = (
        "Mantener `linear` como receta principal y promover el modelo GPU solo cuando mejore el MAPE del mejor baseline clásico en ese SKU. "
        "El backend queda preparado para mezclar ambos mundos sin forzar una sustitución global."
    )
    summary["artifacts"] = {
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "metrics_summary": str(METRICS_SUMMARY_PATH.relative_to(ROOT)),
        "experiment_report": str(EXPERIMENT_REPORT_PATH.relative_to(ROOT)),
        "model_registry": str(MODEL_REGISTRY_PATH.relative_to(ROOT)),
        "predictions": str(PREDICTIONS_PATH.relative_to(ROOT)),
        "models_dir": str(MODELS_DIR.relative_to(ROOT)),
    }

    write_json(
        CONFIG_PATH,
        {
            "model_name": MODEL_NAME,
            "hyperparameter_grid": HYPERPARAM_GRID,
            "feature_columns": FEATURE_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
    )
    write_json(METRICS_SUMMARY_PATH, summary)
    write_json(MODEL_REGISTRY_PATH, registry)
    pd.DataFrame(prediction_rows).to_csv(PREDICTIONS_PATH, index=False)
    EXPERIMENT_REPORT_PATH.write_text(report_markdown(summary), encoding="utf-8")

    return {
        "config": CONFIG_PATH,
        "metrics_summary": METRICS_SUMMARY_PATH,
        "experiment_report": EXPERIMENT_REPORT_PATH,
        "model_registry": MODEL_REGISTRY_PATH,
        "predictions": PREDICTIONS_PATH,
        "models_dir": MODELS_DIR,
    }


if __name__ == "__main__":
    outputs = run_gpu_training()
    for label, path in outputs.items():
        print(f"{label}: {path}")
