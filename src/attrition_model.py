"""Train an employee attrition classifier with a reproducible scikit-learn pipeline.

This script complements the original R Markdown statistical learning analysis with
an application-friendly Python workflow: preprocessing, model comparison,
minority-class metrics, and model persistence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
TARGET = "Attrition"
RANDOM_STATE = 42


# load data
def load_attrition_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the HR dataset and split features from the Attrition target.

    Args:
        csv_path: Path to the CSV file containing the dataset.

    Returns:
        A tuple containing:
        - pd.DataFrame: Features (X).
        - pd.Series: Target labels (y) encoded as 0/1.
    """
    data = pd.read_csv(csv_path)

    # Remove columns that do not contribute to prediction or leak information
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])

    # Convert 'No'/'Yes' to binary 0/1 for modeling
    y = data[TARGET].map({"No": 0, "Yes": 1})
    x = data.drop(columns=[TARGET])

    return x, y


# preprocess
def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical HR features.

    Args:
        x: The feature dataframe used to determine column types.

    Returns:
        A ColumnTransformer ready to scale numerics and one-hot encode categoricals.
    """
    # Dynamically identify numeric and categorical columns
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            # Scale numeric features to have mean=0 and variance=1
            ("numeric", StandardScaler(), numeric_features),
            # Encode categorical features as one-hot arrays; ignore unknowns during transform
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def candidate_models(class_weight: str | None = "balanced") -> dict[str, object]:
    """Return baseline models that are easy to explain to business stakeholders.

    Args:
        class_weight: Strategy to handle class imbalance, such as 'balanced'.

    Returns:
        A dictionary mapping model names to instantiated scikit-learn estimators.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # GradientBoostingClassifier does not support class_weight natively
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_evaluate(csv_path: Path, model_output: Path) -> pd.DataFrame:
    """Train candidate models, print minority-class metrics, and save the best model.

    Args:
        csv_path: Path to the dataset CSV.
        model_output: Path where the best trained pipeline should be saved.

    Returns:
        A DataFrame containing the performance metrics of all evaluated models.
    """
    x, y = load_attrition_data(csv_path)

    # Split the data while maintaining class distribution
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(x_train)
    results: list[dict[str, float | str]] = []
    fitted_pipelines: dict[str, Pipeline] = {}

    for model_name, estimator in candidate_models().items():
        # Create a pipeline combining preprocessing and the model
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        # train models
        # Fit on training data
        pipeline.fit(x_train, y_train)

        # Predict classes and probabilities on test data
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]

        # evaluate models
        # Evaluate using classification report
        report = classification_report(
            y_test,
            predicted,
            output_dict=True,
            zero_division=0,
        )

        results.append(
            {
                "model": model_name,
                "accuracy": report["accuracy"],
                "attrition_precision": report["1"]["precision"],
                "attrition_recall": report["1"]["recall"],
                "attrition_f1": report["1"]["f1-score"],
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )

        fitted_pipelines[model_name] = pipeline

        # Print confusion matrix and detailed report for each model
        print(f"\n{model_name}")
        print(confusion_matrix(y_test, predicted))
        print(classification_report(y_test, predicted, zero_division=0))

    # choose best model
    # Sort results to identify the best model
    # Business priority: recall for attrition class, then ROC AUC
    results_df = pd.DataFrame(results).sort_values(
        ["attrition_recall", "roc_auc"],
        ascending=False,
    )

    # save model
    # Save the best model and model comparison metrics to disk
    best_model_name = str(results_df.iloc[0]["model"])
    model_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(fitted_pipelines[best_model_name], model_output)

    # save metrics
    metrics_output = model_output.parent / "model_metrics.csv"
    results_df.to_csv(metrics_output, index=False)

    print("\nModel comparison")
    print(results_df.to_string(index=False))
    print(f"\nSaved best model ({best_model_name}) to {model_output}")
    print(f"Saved model metrics to {metrics_output}")

    return results_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train employee attrition models.")

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/HR_Employee_Attrition.csv"),
        help="Path to the dataset.",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/attrition_pipeline.joblib"),
        help="Output path for the best model pipeline.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.data, args.model_output)