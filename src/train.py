
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "heart_disease_raw.csv"
    models_dir = project_root / "models"
    results_dir = project_root / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    mlflow_db = project_root / "mlflow.db"

    mlflow_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(
            "sqlite:///" + mlflow_db.as_posix()
    )
    mlflow.set_experiment("Heart Disease Prediction")

    # Continue with your existing code
    df = pd.read_csv(data_path)
    # Load original data and define binary target.
    df = pd.read_csv(data_path)
    df["target"] = (df["num"] > 0).astype(int)
    # Do not use num: it directly determines target.
    X = df.drop(columns=["num", "target"])
    y = df["target"]

    # Treat coded categories as categorical rather than continuous numbers.
    categorical_features = [
        "sex", "cp", "fbs", "restecg",
        "exang", "slope", "ca", "thal",
    ]
    numeric_features = [
        "age", "trestbps", "chol", "thalach", "oldpeak",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        ),
    }

    # Keep a separate test set for final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = []
    trained_pipelines = {}

    print("=" * 60)
    print("HEART DISEASE MODEL TRAINING")
    print("=" * 60)
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        print(f"\nEvaluating {name} with 5-fold cross-validation...")

        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )
        cv_summary = {
            metric: cv_results[f"test_{metric}"].mean()
            for metric in scoring
        }
        # Fit only on the training split before evaluating the test split.
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "test_recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
        }
        # Log experiment details to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model_name": name,
                "cv_folds": 5,
                "test_size": 0.20,
                "random_state": 42,
            })

            # Log cross-validation and test metrics
            mlflow.log_metrics({
                **{
                    f"cv_{key}": float(value)
                    for key, value in cv_summary.items()
                },
                **{
                    key: float(value)
                    for key, value in test_metrics.items()
                },
            })

            # Save and log the classification report
            report_text = classification_report(
                y_test,
                predictions,
                target_names=["No disease", "Disease present"],
                zero_division=0,
            )

            report_path = results_dir / (
                name.lower().replace(" ", "_") + "_report.txt"
            )
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

            # Create and log the confusion matrix
            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=["No disease", "Disease present"],
                ax=ax,
            )
            ax.set_title(f"{name} - Confusion Matrix")

            plot_path = results_dir / (
                name.lower().replace(" ", "_") + "_confusion_matrix.png"
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            mlflow.log_artifact(str(plot_path))

            # Log the complete fitted pipeline

            mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                )
        print(f"MLflow logging completed for {name}")

        results.append({
            "model": name,
            **{f"cv_{k}": v for k, v in cv_summary.items()},
            **test_metrics,
        })
        trained_pipelines[name] = pipeline

        print("\nCross-validation mean:")
        for metric, value in cv_summary.items():
            print(f"  {metric}: {value:.3f}")

        print("\nHeld-out test metrics:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.3f}")

        print("\nClassification report:")
        print(classification_report(
            y_test, predictions,
            target_names=["No disease", "Disease present"],
            zero_division=0,
        ))

        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))

    # Select the model using cross-validation ROC-AUC, not test results.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "cv_roc_auc", ascending=False
    )
    results_df.to_csv(results_dir / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[best_name]

    model_path = models_dir / "heart_disease_pipeline.joblib"
    joblib.dump(best_pipeline, model_path)

    print("\n" + "=" * 60)
    print(f"Selected model: {best_name}")
    print("Selection criterion: highest mean CV ROC-AUC")
    print(f"Saved model pipeline to: {model_path}")
    print(f"Saved comparison results to: {results_dir / 'model_comparison.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
