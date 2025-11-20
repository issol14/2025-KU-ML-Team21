#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Regressors
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None


def try_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def detect_id_columns(df: pd.DataFrame, user_ids: Optional[List[str]]) -> List[str]:
    if user_ids:
        return [c for c in user_ids if c in df.columns]
    candidates = []
    lower_map = {c.lower(): c for c in df.columns}
    for key in ("id", "index"):
        if key in lower_map:
            candidates.append(lower_map[key])
    return candidates


def split_features_targets(
    df: pd.DataFrame,
    label_columns: List[str],
    id_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_labels = [c for c in label_columns if c not in df.columns]
    if missing_labels:
        raise ValueError(f"레이블 컬럼을 찾을 수 없습니다: {missing_labels}")
    exclude_columns = set(label_columns)
    if id_columns:
        exclude_columns.update(id_columns)
    feature_columns = [c for c in df.columns if c not in exclude_columns]
    X = df[feature_columns].copy()
    y = df[label_columns].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    X_processed = X.copy()
    for col in X_processed.select_dtypes(include=["bool"]).columns:
        X_processed[col] = X_processed[col].astype("int64")
    numeric_cols = X_processed.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = [c for c in X_processed.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def available_models(random_state: int) -> Dict[str, object]:
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "Ridge(alpha=1.0)": Ridge(alpha=1.0),
        "Lasso(alpha=0.01)": Lasso(alpha=0.01, max_iter=10000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
    }
    try:
        from xgboost import XGBRegressor  # type: ignore

        models["XGBRegressor"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    except Exception:
        pass
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    target_count = y_true.shape[1]
    r2_list, rmse_list, mae_list = [], [], []
    for i in range(target_count):
        r2_list.append(r2_score(y_true[:, i], y_pred[:, i]))
        rmse_list.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae_list.append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics[f"R2_{label_names[i]}"] = r2_list[-1]
        metrics[f"RMSE_{label_names[i]}"] = rmse_list[-1]
        metrics[f"MAE_{label_names[i]}"] = mae_list[-1]
    metrics["R2_mean"] = float(np.mean(r2_list))
    metrics["RMSE_mean"] = float(np.mean(rmse_list))
    metrics["MAE_mean"] = float(np.mean(mae_list))
    return metrics


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    default_csv = default_root / "ml_feature_data.csv"

    parser = argparse.ArgumentParser(description="다중 타깃 회귀 모델 비교 학습 스크립트")
    parser.add_argument("--csv", type=str, default=str(default_csv), help="학습용 CSV 경로")
    parser.add_argument("--labels", type=str, nargs="*", default=["승차", "하차"], help="레이블 컬럼들")
    parser.add_argument("--id-cols", type=str, nargs="*", default=["ID"], help="제외할 ID 컬럼들")
    parser.add_argument("--test-size", type=float, default=0.2, help="테스트 셋 비율")
    parser.add_argument("--random-state", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--outdir", type=str, default=str(default_root / "preprocess" / "models"), help="모델/리포트 출력 디렉터리")

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

    df = try_read_csv(csv_path)
    id_columns = detect_id_columns(df, args.id_cols)
    X, y = split_features_targets(df, label_columns=args.labels, id_columns=id_columns)

    label_names = list(y.columns)
    y_values = y.values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_values, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor = build_preprocessor(X_train)

    models = available_models(args.random_state)
    results: List[Dict[str, float]] = []

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    for name, estimator in models.items():
        wrapped = MultiOutputRegressor(estimator)
        model = Pipeline(steps=[("preprocess", preprocessor), ("model", wrapped)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred, label_names)
        metrics_with_name: Dict[str, float] = {"model": name}  # type: ignore
        metrics_with_name.update(metrics)
        results.append(metrics_with_name)

        if joblib is not None:
            model_path = outdir / f"{name.replace(' ', '_').replace('(', '[').replace(')', ']')}_pipeline.joblib"
            try:
                joblib.dump(model, model_path)
            except Exception:
                pass

    report = pd.DataFrame(results)
    report_cols = ["model", "R2_mean", "MAE_mean", "RMSE_mean"] + [
        c for c in report.columns if c not in {"model", "R2_mean", "MAE_mean", "RMSE_mean"}
    ]
    report = report[report_cols]
    report_path = outdir / "regression_report.csv"
    try:
        report.to_csv(report_path, index=False, encoding="utf-8")
    except Exception:
        report.to_csv(report_path, index=False)

    best_idx = int(report["R2_mean"].astype(float).idxmax())
    best_row = report.iloc[best_idx]
    print("모델 비교 결과 요약 (상위 지표는 R2_mean 기준):")
    print(report.sort_values("R2_mean", ascending=False).to_string(index=False))
    print("\n최고 모델:")
    print(best_row.to_string())
    print(f"\n리포트 저장: {report_path}")
    print(f"모델 저장 경로: {outdir}")


if __name__ == "__main__":
    main()




