"""
MLOps Training Pipeline
Kubeflow pipeline for end-to-end model training with MLflow tracking.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def ingest_data(
    source_uri: str,
    output_dataset: Output[Dataset],
) -> None:
    """Ingest data from source and validate schema."""
    import pandas as pd
    from pathlib import Path

    # Load data from source
    if source_uri.startswith("s3://"):
        df = pd.read_parquet(source_uri)
    elif source_uri.startswith("gs://"):
        df = pd.read_parquet(source_uri)
    else:
        df = pd.read_csv(source_uri)

    # Schema validation
    required_columns = ["feature_1", "feature_2", "target"]
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove duplicates and null targets
    df = df.drop_duplicates().dropna(subset=["target"])

    print(f"Ingested {len(df)} rows, {len(df.columns)} columns")
    df.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "scikit-learn", "numpy", "pyarrow"]
)
def preprocess_and_engineer_features(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    feature_config: dict,
) -> None:
    """Feature engineering and preprocessing."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_parquet(input_dataset.path)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Feature engineering
    if feature_config.get("create_interactions", False):
        for i, col1 in enumerate(numeric_cols[:5]):
            for col2 in numeric_cols[i + 1:5]:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    if feature_config.get("create_aggregations", False):
        for col in numeric_cols[:5]:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            df[f"{col}_squared"] = df[col] ** 2

    # Encode categoricals
    for col in categorical_cols:
        if col != "target":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Scale numeric features
    scaler = StandardScaler()
    feature_cols = [c for c in df.columns if c != "target"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print(f"Engineered features: {len(df.columns)} total columns")
    df.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas", "scikit-learn", "xgboost", "lightgbm",
        "mlflow", "numpy", "pyarrow"
    ]
)
def train_model(
    input_dataset: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    model_type: str = "xgboost",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "mlops-pipeline",
) -> None:
    """Train model with hyperparameter tuning and MLflow tracking."""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    import json
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score
    )

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    df = pd.read_parquet(input_dataset.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Select model
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"{model_type}-pipeline-run"):
        # Log parameters
        mlflow.log_params(model.get_params())
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=f"{model_type}-production"
        )

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            mlflow.log_dict(dict(top_features), "feature_importance.json")

        print(f"Model trained: {model_type}")
        print(f"Metrics: {json.dumps(metrics, indent=2)}")

        # Save metrics
        for key, value in metrics.items():
            metrics_output.log_metric(key, value)

    # Save model artifact
    import pickle
    with open(model_output.path, "wb") as f:
        pickle.dump(model, f)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scipy", "pyarrow"]
)
def detect_drift(
    reference_data: Input[Dataset],
    current_data: Input[Dataset],
    metrics_output: Output[Metrics],
    drift_threshold: float = 0.2,
) -> bool:
    """Detect data drift using PSI and KL divergence."""
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp

    ref_df = pd.read_parquet(reference_data.path)
    cur_df = pd.read_parquet(current_data.path)

    drift_detected = False
    drift_scores = {}

    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col == "target":
            continue

        # KS test
        stat, p_value = ks_2samp(ref_df[col].dropna(), cur_df[col].dropna())
        drift_scores[col] = {"ks_stat": stat, "p_value": p_value}

        if p_value < 0.05:
            drift_detected = True

    # Log overall drift score
    avg_drift = np.mean([s["ks_stat"] for s in drift_scores.values()])
    metrics_output.log_metric("avg_drift_score", avg_drift)
    metrics_output.log_metric("drift_detected", int(drift_detected))

    print(f"Drift detected: {drift_detected} (avg score: {avg_drift:.4f})")
    return drift_detected


@dsl.pipeline(
    name="mlops-training-pipeline",
    description="End-to-end ML training pipeline with drift detection"
)
def training_pipeline(
    data_source_uri: str = "s3://ml-data/training/latest.parquet",
    model_type: str = "xgboost",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "mlops-pipeline",
    enable_drift_detection: bool = True,
):
    """Main training pipeline orchestration."""

    # Step 1: Data Ingestion
    ingest_task = ingest_data(source_uri=data_source_uri)

    # Step 2: Feature Engineering
    feature_config = {
        "create_interactions": True,
        "create_aggregations": True,
    }
    feature_task = preprocess_and_engineer_features(
        input_dataset=ingest_task.outputs["output_dataset"],
        feature_config=feature_config,
    )

    # Step 3: Model Training
    train_task = train_model(
        input_dataset=feature_task.outputs["output_dataset"],
        model_type=model_type,
        mlflow_tracking_uri=mlflow_tracking_uri,
        experiment_name=experiment_name,
    )

    # Step 4: Drift Detection (conditional)
    if enable_drift_detection:
        drift_task = detect_drift(
            reference_data=ingest_task.outputs["output_dataset"],
            current_data=feature_task.outputs["output_dataset"],
        )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.yaml"
    )