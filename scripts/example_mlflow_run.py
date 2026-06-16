"""
Example MLflow training run.

TODO: delete before delivery if Scott has not picked up MLflow integration.
Stub from 2026-04-08; the project settled on Aim + run_tracker for experiment
tracking (see ``src/bst_x/run_tracker.md``). Kept around because Scott
may still wire MLflow in for Architecture 2.

Demonstrates how to log experiments with MLflow. Run this script to verify
MLflow is working, then view results with:

    mlflow ui

and open http://127.0.0.1:5000 in your browser.
"""

import mlflow
import math

# --- Example hyperparameters (replace with real ones during training) ---
HYPERPARAMS = {
    "model": "Model B (keypoint)",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "num_classes": 14,
}

with mlflow.start_run(run_name="example_run"):
    # Log hyperparameters
    mlflow.log_params(HYPERPARAMS)

    # Simulate a loss curve over epochs
    for epoch in range(1, HYPERPARAMS["epochs"] + 1):
        fake_loss = 1.0 / math.sqrt(epoch)
        fake_accuracy = 1.0 - fake_loss * 0.5
        mlflow.log_metric("train_loss", fake_loss, step=epoch)
        mlflow.log_metric("train_accuracy", fake_accuracy, step=epoch)

    # Log final summary metrics
    mlflow.log_metric("final_accuracy", 0.87)
    mlflow.log_metric("final_loss", 0.32)

    print("Run logged. View results with: mlflow ui")
