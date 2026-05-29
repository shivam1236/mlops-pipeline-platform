"""
MLOps Retraining Pipeline
Automated retraining triggered by drift detection or scheduled intervals.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scipy", "pyarrow"]
)
def check_drift_trigger(
    reference_uri: str,
    current_uri: str,
    metrics_output: Output[Metrics],
    drift_threshold: float = 0.05,
) -> bool:
    """Check if model retraining should be triggered based on data drift."""
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp

    ref_df = pd.read_parquet(reference_uri)
    cur_df = pd.read_parquet(current_uri)

    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
    drifted_features = 0
    total_features = 0

    for col in numeric_cols:
        if col == "target" or col not in cur_df.columns:
            continue
        total_features += 1

        stat, p_value = ks_2samp(
            ref_df[col].dropna(),
            cur_df[col].dropna()
        )
        if p_value < drift_threshold:
            drifted_features += 1

    drift_ratio = drifted_features / max(total_features, 1)
    should_retrain = drift_ratio > 0.3  # Retrain if >30% features drifted

    metrics_output.log_metric("total_features", total_features)
    metrics_output.log_metric("drifted_features", drifted_features)
    metrics_output.log_metric("drift_ratio", drift_ratio)
    metrics_output.log_metric("should_retrain", int(should_retrain))

    print(f"Drift check: {drifted_features}/{total_features} features drifted ({drift_ratio:.1%})")
    print(f"Retrain triggered: {should_retrain}")

    return should_retrain


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "pyarrow"]
)
def prepare_retraining_data(
    historical_uri: str,
    recent_uri: str,
    output_dataset: Output[Dataset],
    recent_weight: float = 0.7,
) -> None:
    """Combine historical and recent data with weighting for retraining."""
    import pandas as pd
    import numpy as np

    historical = pd.read_parquet(historical_uri)
    recent = pd.read_parquet(recent_uri)

    # Sample historical data proportionally
    total_target = int(len(recent) / recent_weight * (1 - recent_weight))
    if len(historical) > total_target:
        historical_sample = historical.sample(n=total_target, random_state=42)
    else:
        historical_sample = historical

    # Combine with priority to recent data
    combined = pd.concat([historical_sample, recent], ignore_index=True)
    combined = combined.drop_duplicates()
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Retraining dataset: {len(combined)} rows")
    print(f"  Historical: {len(historical_sample)} ({(1-recent_weight)*100:.0f}%)")
    print(f"  Recent: {len(recent)} ({recent_weight*100:.0f}%)")

    combined.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas", "numpy", "scikit-learn", "xgboost",
        "lightgbm", "mlflow", "pyarrow"
    ]
)
def retrain_model(
    input_dataset: Input[Dataset],
    model_output: Output[Model],
    metrics_output: Output[Metrics],
    model_type: str = "xgboost",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    experiment_name: str = "mlops-retraining",
) -> None:
    """Retrain model on updated data with comparison to current production."""
    import pandas as pd
    import numpy as np
    import mlflow
    import mlflow.sklearn
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import pickle

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    df = pd.read_parquet(input_dataset.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbose=-1,
        )

    with mlflow.start_run(run_name=f"retrain-{model_type}"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted")
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_param("retrain_reason", "drift_detected")
        mlflow.log_param("training_data_size", len(X_train))

        # Register new model version
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name=f"{model_type}-production"
        )

        for key, value in metrics.items():
            metrics_output.log_metric(key, value)

        print(f"Retrained model metrics: {metrics}")

    with open(model_output.path, "wb") as f:
        pickle.dump(model, f)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "mlflow", "pyarrow"]
)
def compare_with_production(
    test_dataset: Input[Dataset],
    new_model: Input[Model],
    metrics_output: Output[Metrics],
    model_name: str = "xgboost-production",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    min_improvement: float = 0.01,
) -> bool:
    """Compare retrained model against current production model."""
    import pandas as pd
    import numpy as np
    import mlflow
    import pickle
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Load test data
    df = pd.read_parquet(test_dataset.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Load new model
    with open(new_model.path, "rb") as f:
        new = pickle.load(f)

    # Load production model
    try:
        prod_model_uri = f"models:/{model_name}/Production"
        prod = mlflow.sklearn.load_model(prod_model_uri)
        has_production = True
    except Exception:
        print("No production model found. Promoting new model.")
        has_production = False

    # Evaluate new model
    new_preds = new.predict(X)
    new_f1 = f1_score(y, new_preds, average="weighted")
    new_acc = accuracy_score(y, new_preds)

    metrics_output.log_metric("new_model_f1", new_f1)
    metrics_output.log_metric("new_model_accuracy", new_acc)

    if not has_production:
        metrics_output.log_metric("should_promote", 1)
        return True

    # Evaluate production model
    prod_preds = prod.predict(X)
    prod_f1 = f1_score(y, prod_preds, average="weighted")
    prod_acc = accuracy_score(y, prod_preds)

    metrics_output.log_metric("prod_model_f1", prod_f1)
    metrics_output.log_metric("prod_model_accuracy", prod_acc)
    metrics_output.log_metric("f1_improvement", new_f1 - prod_f1)

    # Promote if improved
    should_promote = (new_f1 - prod_f1) >= min_improvement
    metrics_output.log_metric("should_promote", int(should_promote))

    print(f"Production F1: {prod_f1:.4f} | New F1: {new_f1:.4f}")
    print(f"Improvement: {new_f1 - prod_f1:.4f} (threshold: {min_improvement})")
    print(f"Promote: {should_promote}")

    return should_promote


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["mlflow"]
)
def promote_model(
    model_name: str = "xgboost-production",
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> None:
    """Promote latest model version to Production stage."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # Get latest version
    versions = client.get_latest_versions(model_name, stages=["None"])
    if not versions:
        print("No new model version to promote")
        return

    latest = versions[0]

    # Transition current production to archived
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for pv in prod_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=pv.version,
            stage="Archived",
        )

    # Promote new model
    client.transition_model_version_stage(
        name=model_name,
        version=latest.version,
        stage="Production",
    )

    print(f"Promoted model {model_name} v{latest.version} to Production")


@dsl.pipeline(
    name="mlops-retraining-pipeline",
    description="Automated model retraining triggered by drift detection"
)
def retraining_pipeline(
    reference_data_uri: str = "s3://ml-data/reference/baseline.parquet",
    current_data_uri: str = "s3://ml-data/production/recent.parquet",
    historical_data_uri: str = "s3://ml-data/training/historical.parquet",
    model_type: str = "xgboost",
    model_name: str = "xgboost-production",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    drift_threshold: float = 0.05,
    min_improvement: float = 0.01,
):
    """Retraining pipeline: detect drift → retrain → compare → promote."""

    # Step 1: Check if retraining is needed
    drift_task = check_drift_trigger(
        reference_uri=reference_data_uri,
        current_uri=current_data_uri,
        drift_threshold=drift_threshold,
    )

    # Only proceed if drift detected
    with dsl.Condition(drift_task.output == True):

        # Step 2: Prepare combined training data
        data_task = prepare_retraining_data(
            historical_uri=historical_data_uri,
            recent_uri=current_data_uri,
        )

        # Step 3: Retrain model
        retrain_task = retrain_model(
            input_dataset=data_task.outputs["output_dataset"],
            model_type=model_type,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        # Step 4: Compare with production
        compare_task = compare_with_production(
            test_dataset=data_task.outputs["output_dataset"],
            new_model=retrain_task.outputs["model_output"],
            model_name=model_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            min_improvement=min_improvement,
        )

        # Step 5: Promote if better
        with dsl.Condition(compare_task.output == True):
            promote_model(
                model_name=model_name,
                mlflow_tracking_uri=mlflow_tracking_uri,
            )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=retraining_pipeline,
        package_path="retraining_pipeline.yaml"
    )
