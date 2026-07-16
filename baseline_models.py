"""
Prepare the two human sleep datasets and train transparent baseline models.

Run:
    python baseline_models.py

Outputs are written to ./outputs. The animal dataset is intentionally excluded.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def save_prepared_data(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Remove duplicate observations and save a clean, analysis-ready copy."""
    before = len(frame)
    cleaned = frame.drop_duplicates().copy()
    cleaned.to_csv(OUTPUT_DIR / f"{name}_prepared.csv", index=False)
    print(f"{name}: {before} rows -> {len(cleaned)} after exact duplicate removal")
    return cleaned


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def regression_models(frame: pd.DataFrame, target: str, drop_columns: list[str], label: str) -> list[dict]:
    """Compare a mean-prediction baseline with a random-forest baseline."""
    data = frame.dropna(subset=[target]).copy()
    x = data.drop(columns=[target, *drop_columns], errors="ignore")
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )

    models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    results = []
    for model_name, model in models.items():
        pipeline = Pipeline([("prepare", make_preprocessor(x)), ("model", model)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        results.append(
            {
                "dataset": label,
                "task": "regression",
                "target": target,
                "model": model_name,
                "test_rows": len(y_test),
                "MAE": mean_absolute_error(y_test, prediction),
                "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                "R2": r2_score(y_test, prediction),
            }
        )
    return results


def classification_models(
    frame: pd.DataFrame, target: str, drop_columns: list[str], label: str
) -> list[dict]:
    """Compare a majority-class baseline with a random-forest classifier."""
    data = frame.dropna(subset=[target]).copy()
    x = data.drop(columns=[target, *drop_columns], errors="ignore")
    y = data[target].astype(str)

    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )
    models = {
        "dummy_majority_class": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    results = []
    for model_name, model in models.items():
        pipeline = Pipeline([("prepare", make_preprocessor(x)), ("model", model)])
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        results.append(
            {
                "dataset": label,
                "task": "classification",
                "target": target,
                "model": model_name,
                "test_rows": len(y_test),
                "accuracy": accuracy_score(y_test, prediction),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "f1_macro": f1_score(y_test, prediction, average="macro", zero_division=0),
            }
        )
    return results


def plot_main_dataset(frame: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plot_data = frame.dropna(subset=["Sleep Duration", "Stress Level"]).copy()
    plot_data["Sleep category"] = pd.cut(
        plot_data["Sleep Duration"],
        bins=[0, 7, 8, float("inf")],
        labels=["Less than 7 hours", "7–8 hours", "More than 8 hours"],
        right=False,
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_data, x="Sleep category", y="Stress Level", errorbar=None)
    plt.title("Average stress level by sleep-duration group")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stress_by_sleep_duration.png", dpi=200)
    plt.close()


def prepare_efficiency_dataset() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "Sleep_Efficiency.csv")
    data.columns = [column.strip() for column in data.columns]

    # Times are converted to clock minutes, retaining useful timing information
    # without treating the raw timestamps as arbitrary text.
    for source, destination in [("Bedtime", "Bedtime clock minutes"), ("Wakeup time", "Wakeup clock minutes")]:
        parsed = pd.to_datetime(data[source], errors="coerce")
        data[destination] = parsed.dt.hour * 60 + parsed.dt.minute
    return save_prepared_data(data, "sleep_efficiency")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = []

    # keep_default_na=False avoids pandas silently reading the literal string
    # "None" in the Sleep Disorder column as a missing value (it is one of
    # pandas' default NA markers), which would wrongly drop the majority class.
    lifestyle = pd.read_csv(
        ROOT / "Sleep_health_and_lifestyle_dataset.csv", keep_default_na=False, na_values=[""]
    )
    lifestyle.columns = [column.strip() for column in lifestyle.columns]
    lifestyle = save_prepared_data(lifestyle.drop(columns=["Person ID"]), "sleep_health_lifestyle")
    plot_main_dataset(lifestyle)

    # Main biomedical questions: stress, sleep quality, and sleep-disorder status.
    all_results.extend(
        regression_models(
            lifestyle, "Stress Level", [], "sleep_health_lifestyle"
        )
    )
    all_results.extend(
        regression_models(
            lifestyle, "Quality of Sleep", [], "sleep_health_lifestyle"
        )
    )
    all_results.extend(
        classification_models(
            lifestyle, "Sleep Disorder", [], "sleep_health_lifestyle"
        )
    )

    efficiency = prepare_efficiency_dataset()
    # Avoid allowing the record ID or original timestamp text to act as features.
    timestamp_columns = ["ID", "Bedtime", "Wakeup time"]
    all_results.extend(
        regression_models(
            efficiency, "Sleep efficiency", timestamp_columns, "sleep_efficiency"
        )
    )
    all_results.extend(
        regression_models(
            efficiency, "Awakenings", timestamp_columns, "sleep_efficiency"
        )
    )

    results = pd.DataFrame(all_results)
    results.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False)
    with open(OUTPUT_DIR / "model_notes.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "75/25 train-test split, random state 42",
                "regression_metrics": "Lower MAE/RMSE is better; higher R2 is better.",
                "classification_metrics": "Accuracy, balanced accuracy, and macro F1 are higher-is-better.",
                "important_limit": (
                    "These public, observational datasets show association only. "
                    "They do not establish that one health factor causes another."
                ),
            },
            file,
            indent=2,
        )
    print("\nSaved prepared data, a figure, and model metrics in:", OUTPUT_DIR)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
