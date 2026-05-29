"""
MLOps Inference Pipeline
Kubeflow pipeline for batch inference with monitoring and validation.
"""

from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "pyarrow"]
)
def load_inference_data(
    source_uri: str,
    output_dataset: Output[Dataset],
) -> None:
    """Load new data for batch inference."""
    import pandas as pd

    if source_uri.startswith("s3://") or source_uri.startswith("gs://"):
        df = pd.read_parquet(source_uri)
    else:
        df = pd.read_csv(source_uri)

    # Validate input schema
    if df.empty:
        raise ValueError("Input data is empty")

    print(f"Loaded {len(df)} rows for inference")
    df.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "scikit-learn", "pyarrow"]
)
def preprocess_inference_data(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    feature_columns: list,
) -> None:
    """Preprocess data using the same transformations as training."""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    df = pd.read_parquet(input_dataset.path)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    # Encode categoricals
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Scale features
    scaler = StandardScaler()
    if feature_columns:
        cols_to_scale = [c for c in feature_columns if c in df.columns]
    else:
        cols_to_scale = list(numeric_cols)

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print(f"Preprocessed {len(df)} rows, {len(df.columns)} features")
    df.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "mlflow", "scikit-learn", "pyarrow"]
)
def run_batch_inference(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    metrics_output: Output[Metrics],
    model_name: str = "xgboost-production",
    model_stage: str = "Production",
    mlflow_tracking_uri: str = "http://mlflow:5000",
) -> None:
    """Run batch predictions using the production model from MLflow registry."""
    import pandas as pd
    import numpy as np
    import mlflow
    import time

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Load model from registry
    model_uri = f"models:/{model_name}/{model_stage}"
    print(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

    # Load data
    df = pd.read_parquet(input_dataset.path)

    # Run inference
    start_time = time.time()
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)
    inference_time = time.time() - start_time

    # Attach predictions
    df["prediction"] = predictions
    df["confidence"] = np.max(probabilities, axis=1)
    df["prediction_timestamp"] = pd.Timestamp.now().isoformat()

    # Log metrics
    avg_confidence = float(np.mean(df["confidence"]))
    low_confidence_pct = float((df["confidence"] < 0.7).mean() * 100)
    latency_per_row = (inference_time / len(df)) * 1000  # ms

    metrics_output.log_metric("total_predictions", len(df))
    metrics_output.log_metric("avg_confidence", avg_confidence)
    metrics_output.log_metric("low_confidence_pct", low_confidence_pct)
    metrics_output.log_metric("inference_time_seconds", inference_time)
    metrics_output.log_metric("latency_per_row_ms", latency_per_row)

    print(f"Inference complete: {len(df)} predictions in {inference_time:.2f}s")
    print(f"Avg confidence: {avg_confidence:.4f}, Low confidence: {low_confidence_pct:.1f}%")

    df.to_parquet(output_dataset.path, index=False)


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "numpy", "pyarrow"]
)
def validate_predictions(
    predictions_dataset: Input[Dataset],
    metrics_output: Output[Metrics],
    min_confidence: float = 0.5,
    max_null_predictions_pct: float = 1.0,
) -> bool:
    """Validate prediction quality before writing to output."""
    import pandas as pd
    import numpy as np

    df = pd.read_parquet(predictions_dataset.path)

    # Validation checks
    null_predictions = df["prediction"].isnull().sum()
    null_pct = (null_predictions / len(df)) * 100
    low_confidence = (df["confidence"] < min_confidence).sum()
    low_conf_pct = (low_confidence / len(df)) * 100

    # Class distribution
    class_distribution = df["prediction"].value_counts(normalize=True).to_dict()

    metrics_output.log_metric("null_predictions_pct", null_pct)
    metrics_output.log_metric("low_confidence_pct", low_conf_pct)

    # Check for anomalies
    is_valid = True
    issues = []

    if null_pct > max_null_predictions_pct:
        issues.append(f"Too many null predictions: {null_pct:.1f}%")
        is_valid = False

    if low_conf_pct > 30:
        issues.append(f"Too many low-confidence predictions: {low_conf_pct:.1f}%")
        is_valid = False

    # Check for class imbalance anomaly
    for cls, pct in class_distribution.items():
        if pct > 0.95:
            issues.append(f"Class {cls} dominates at {pct*100:.1f}%")
            is_valid = False

    if issues:
        print(f"Validation issues: {issues}")
    else:
        print("All validation checks passed")

    return is_valid


@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=["pandas", "pyarrow"]
)
def write_predictions(
    predictions_dataset: Input[Dataset],
    output_uri: str,
    output_format: str = "parquet",
) -> None:
    """Write validated predictions to output storage."""
    import pandas as pd

    df = pd.read_parquet(predictions_dataset.path)

    if output_format == "parquet":
        df.to_parquet(output_uri, index=False)
    elif output_format == "csv":
        df.to_csv(output_uri, index=False)
    else:
        raise ValueError(f"Unsupported format: {output_format}")

    print(f"Written {len(df)} predictions to {output_uri}")


@dsl.pipeline(
    name="mlops-inference-pipeline",
    description="Batch inference pipeline with validation and monitoring"
)
def inference_pipeline(
    data_source_uri: str = "s3://ml-data/inference/incoming.parquet",
    output_uri: str = "s3://ml-data/predictions/output.parquet",
    model_name: str = "xgboost-production",
    model_stage: str = "Production",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    min_confidence: float = 0.5,
):
    """Batch inference pipeline with quality gates."""

    # Step 1: Load data
    load_task = load_inference_data(source_uri=data_source_uri)

    # Step 2: Preprocess
    preprocess_task = preprocess_inference_data(
        input_dataset=load_task.outputs["output_dataset"],
        feature_columns=[],
    )

    # Step 3: Run inference
    inference_task = run_batch_inference(
        input_dataset=preprocess_task.outputs["output_dataset"],
        model_name=model_name,
        model_stage=model_stage,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    # Step 4: Validate predictions
    validate_task = validate_predictions(
        predictions_dataset=inference_task.outputs["output_dataset"],
        min_confidence=min_confidence,
    )

    # Step 5: Write output (only if valid)
    with dsl.Condition(validate_task.output == True):
        write_predictions(
            predictions_dataset=inference_task.outputs["output_dataset"],
            output_uri=output_uri,
        )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path="inference_pipeline.yaml"
    )