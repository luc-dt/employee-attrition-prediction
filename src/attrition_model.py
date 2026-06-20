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


def load_attrition_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the HR dataset and split features from the Attrition target."""
    data = pd.read_csv(csv_path)
    data = data.drop(columns=[col for col in DROP_COLUMNS if col in data.columns])
    y = data[TARGET].map({"No": 0, "Yes": 1})
    x = data.drop(columns=[TARGET])
    return x, y


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical HR features."""
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = x.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def candidate_models(class_weight: str | None = "balanced") -> dict[str, object]:
    """Return baseline models that are easy to explain to business stakeholders."""
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
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def train_and_evaluate(csv_path: Path, model_output: Path) -> pd.DataFrame:
    """Train candidate models, print minority-class metrics, and save the best model."""
    x, y = load_attrition_data(csv_path)
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
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        report = classification_report(y_test, predicted, output_dict=True, zero_division=0)

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
        print(f"\n{model_name}")
        print(confusion_matrix(y_test, predicted))
        print(classification_report(y_test, predicted, zero_division=0))

    results_df = pd.DataFrame(results).sort_values(
        ["attrition_recall", "roc_auc"], ascending=False
    )
    best_model_name = str(results_df.iloc[0]["model"])
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipelines[best_model_name], model_output)
    print("\nModel comparison")
    print(results_df.to_string(index=False))
    print(f"\nSaved best model ({best_model_name}) to {model_output}")
    return results_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train employee attrition models.")
    parser.add_argument("--data", type=Path, default=Path("HR_Employee_Attrition.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("models/attrition_pipeline.joblib"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_and_evaluate(args.data, args.model_output)
