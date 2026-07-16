"""
Sleep, Stress, and Lifestyle Insights - interactive web app.

Combines the baseline models built in baseline_models.py into a single
Streamlit site: data overview, model performance comparisons, an
interactive predictor, and the full written report.

Run locally with:
    streamlit run app.py

Deploy for free at https://share.streamlit.io by connecting this
repository (see README.md for step-by-step instructions).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
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
RANDOM_STATE = 42

st.set_page_config(
    page_title="Sleep, Stress & Lifestyle Insights",
    page_icon="🌙",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_lifestyle() -> pd.DataFrame:
    # keep_default_na=False avoids pandas silently reading the literal string
    # "None" in the Sleep Disorder column as a missing value (it is one of
    # pandas' default NA markers), which would wrongly drop the majority class.
    df = pd.read_csv(ROOT / "Sleep_health_and_lifestyle_dataset.csv", keep_default_na=False, na_values=[""])
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=["Person ID"]).drop_duplicates().reset_index(drop=True)
    split_bp = df["Blood Pressure"].str.split("/", expand=True)
    df["Systolic BP"] = pd.to_numeric(split_bp[0], errors="coerce")
    df["Diastolic BP"] = pd.to_numeric(split_bp[1], errors="coerce")
    return df


@st.cache_data
def load_efficiency() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "Sleep_Efficiency.csv")
    df.columns = [c.strip() for c in df.columns]
    for source, destination in [
        ("Bedtime", "Bedtime clock minutes"),
        ("Wakeup time", "Wakeup clock minutes"),
    ]:
        parsed = pd.to_datetime(df[source], errors="coerce")
        df[destination] = parsed.dt.hour * 60 + parsed.dt.minute
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Modeling helpers (same approach as baseline_models.py)
# ---------------------------------------------------------------------------
def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
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


@st.cache_resource(show_spinner="Training regression models...")
def train_regressors(df: pd.DataFrame, target: str, drop_columns: tuple[str, ...]):
    data = df.dropna(subset=[target]).copy()
    x = data.drop(columns=[target, *drop_columns], errors="ignore")
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE)

    model_specs = {
        "Dummy baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    fitted, rows = {}, []
    for name, model in model_specs.items():
        pipe = Pipeline([("prep", make_preprocessor(x)), ("model", model)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred) ** 0.5,
                "R2": r2_score(y_test, pred),
            }
        )
        fitted[name] = pipe
    return fitted, pd.DataFrame(rows), x.columns.tolist()


@st.cache_resource(show_spinner="Training classification models...")
def train_classifiers(df: pd.DataFrame, target: str, drop_columns: tuple[str, ...]):
    data = df.dropna(subset=[target]).copy()
    x = data.drop(columns=[target, *drop_columns], errors="ignore")
    y = data[target].astype(str)
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE, stratify=stratify
    )

    model_specs = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    fitted, rows = {}, []
    for name, model in model_specs.items():
        pipe = Pipeline([("prep", make_preprocessor(x)), ("model", model)])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
                "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
            }
        )
        fitted[name] = pipe
    classes = sorted(y.unique())
    return fitted, pd.DataFrame(rows), x.columns.tolist(), classes


def feature_importance_chart(pipe: Pipeline, feature_names: list[str], top_n: int = 10):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    prep = pipe.named_steps["prep"]
    numeric = prep.transformers_[0][2]
    cat_encoder = prep.transformers_[1][1].named_steps["one_hot"]
    categorical = prep.transformers_[1][2]
    cat_names = list(cat_encoder.get_feature_names_out(categorical)) if categorical else []
    all_names = list(numeric) + cat_names
    importance = pd.Series(model.feature_importances_, index=all_names).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=importance.values, y=importance.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("🌙 Sleep, Stress & Lifestyle Insights")
    st.caption("A biomedical data-science project exploring sleep, stress, and lifestyle patterns")

    st.markdown(
        """
        This site presents a student research project analyzing two public, de-identified
        human sleep datasets from Kaggle. The goal is to understand how sleep habits and
        lifestyle factors relate to stress, sleep quality, sleep disorders, and sleep
        efficiency — and to turn those findings into practical, evidence-based
        recommendations that students and families (including here in Irvine, CA) can use.

        **Use the sidebar to explore:**
        - **Explore the Data** — charts and summary statistics for both datasets
        - **Model Performance** — how well simple vs. advanced models predict each outcome
        - **Try the Predictor** — enter your own lifestyle info and see a model's prediction
        - **Full Report** — the complete written analysis
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset 1: Sleep Health & Lifestyle")
        st.write(
            "374 records → 132 unique after removing exact duplicate rows. "
            "Includes sleep duration/quality, stress level, activity, BMI, "
            "blood pressure, heart rate, daily steps, and sleep disorder."
        )
    with col2:
        st.subheader("Dataset 2: Sleep Efficiency")
        st.write(
            "452 records. Includes sleep duration/efficiency, sleep-stage "
            "percentages, awakenings, caffeine/alcohol use, smoking, and "
            "exercise frequency."
        )

    st.info(
        "**Important limitation:** these datasets are public and not specific to Irvine. "
        "All results show *associations*, not proof that one factor *causes* another.",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Page: Explore the data
# ---------------------------------------------------------------------------
def page_explore():
    st.title("Explore the Data")
    lifestyle = load_lifestyle()
    efficiency = load_efficiency()

    dataset_choice = st.radio("Dataset", ["Sleep Health & Lifestyle", "Sleep Efficiency"], horizontal=True)

    if dataset_choice == "Sleep Health & Lifestyle":
        df = lifestyle
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Average stress level by sleep-duration group")
        plot_data = df.dropna(subset=["Sleep Duration", "Stress Level"]).copy()
        plot_data["Sleep category"] = pd.cut(
            plot_data["Sleep Duration"],
            bins=[0, 7, 8, float("inf")],
            labels=["Less than 7 hours", "7-8 hours", "More than 8 hours"],
            right=False,
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=plot_data, x="Sleep category", y="Stress Level", errorbar=None, ax=ax, color="#4C72B0")
        ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Sleep quality vs. stress level")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Quality of Sleep", y="Stress Level", hue="Sleep Disorder", ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
    else:
        df = efficiency
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Sleep duration vs. sleep efficiency")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Sleep duration", y="Sleep efficiency", hue="Smoking status", ax=ax)
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Caffeine consumption vs. sleep efficiency")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Caffeine consumption", y="Sleep efficiency", ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model performance
# ---------------------------------------------------------------------------
def page_model_performance():
    st.title("Model Performance")
    st.write(
        "Every target is predicted with three models trained on the same data: a naive "
        "**dummy baseline**, an interpretable **linear/logistic regression**, and a more "
        "flexible **random forest**. Metrics are computed on the same held-out test rows."
    )

    lifestyle = load_lifestyle()
    efficiency = load_efficiency()

    st.header("Sleep Health & Lifestyle dataset")

    st.subheader("Target: Stress Level (0-10)")
    fitted, metrics, features = train_regressors(lifestyle, "Stress Level", ())
    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    fig = feature_importance_chart(fitted["Random Forest"], features)
    if fig:
        st.caption("Top features driving the Random Forest's stress-level predictions")
        st.pyplot(fig)

    st.subheader("Target: Quality of Sleep (1-10)")
    fitted_q, metrics_q, features_q = train_regressors(lifestyle, "Quality of Sleep", ())
    st.dataframe(metrics_q.round(3), use_container_width=True, hide_index=True)
    fig_q = feature_importance_chart(fitted_q["Random Forest"], features_q)
    if fig_q:
        st.caption("Top features driving the Random Forest's sleep-quality predictions")
        st.pyplot(fig_q)

    st.subheader("Target: Sleep Disorder (None / Insomnia / Sleep Apnea)")
    fitted_d, metrics_d, features_d, classes_d = train_classifiers(lifestyle, "Sleep Disorder", ())
    st.dataframe(metrics_d.round(3), use_container_width=True, hide_index=True)
    fig_d = feature_importance_chart(fitted_d["Random Forest"], features_d)
    if fig_d:
        st.caption("Top features driving the Random Forest's sleep-disorder predictions")
        st.pyplot(fig_d)

    st.header("Sleep Efficiency dataset")
    timestamp_cols = ("ID", "Bedtime", "Wakeup time")

    st.subheader("Target: Sleep Efficiency (0-1)")
    fitted_e, metrics_e, features_e = train_regressors(efficiency, "Sleep efficiency", timestamp_cols)
    st.dataframe(metrics_e.round(3), use_container_width=True, hide_index=True)
    fig_e = feature_importance_chart(fitted_e["Random Forest"], features_e)
    if fig_e:
        st.caption("Top features driving the Random Forest's sleep-efficiency predictions")
        st.pyplot(fig_e)

    st.subheader("Target: Awakenings (count per night)")
    fitted_a, metrics_a, features_a = train_regressors(efficiency, "Awakenings", timestamp_cols)
    st.dataframe(metrics_a.round(3), use_container_width=True, hide_index=True)
    fig_a = feature_importance_chart(fitted_a["Random Forest"], features_a)
    if fig_a:
        st.caption("Top features driving the Random Forest's awakenings predictions")
        st.pyplot(fig_a)


# ---------------------------------------------------------------------------
# Page: Interactive predictor
# ---------------------------------------------------------------------------
LIFESTYLE_PREDICTOR_FEATURES = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Physical Activity Level",
    "BMI Category",
    "Systolic BP",
    "Diastolic BP",
    "Heart Rate",
    "Daily Steps",
]

EFFICIENCY_PREDICTOR_FEATURES = [
    "Age",
    "Gender",
    "Sleep duration",
    "Caffeine consumption",
    "Alcohol consumption",
    "Smoking status",
    "Exercise frequency",
]


def page_predictor():
    st.title("Try the Predictor")
    st.warning(
        "This is an educational demo based on public datasets, not a medical tool. "
        "It does not diagnose conditions or replace advice from a doctor.",
        icon="🩺",
    )

    dataset_choice = st.radio(
        "Which model would you like to try?",
        ["Stress, sleep quality & sleep disorder (Lifestyle dataset)", "Sleep efficiency & awakenings (Efficiency dataset)"],
    )

    if dataset_choice.startswith("Stress"):
        lifestyle = load_lifestyle()
        base = lifestyle[LIFESTYLE_PREDICTOR_FEATURES]

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", sorted(base["Gender"].dropna().unique()))
            age = st.slider("Age", 18, 80, 25)
            occupation = st.selectbox("Occupation", sorted(base["Occupation"].dropna().unique()))
        with col2:
            sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, 0.1)
            activity = st.slider("Physical Activity Level (0-100)", 0, 100, 50)
            bmi = st.selectbox("BMI Category", sorted(base["BMI Category"].dropna().unique()))
        with col3:
            systolic = st.slider("Systolic blood pressure", 90, 180, 120)
            diastolic = st.slider("Diastolic blood pressure", 60, 120, 80)
            heart_rate = st.slider("Resting heart rate (bpm)", 50, 110, 72)
            steps = st.slider("Daily steps", 1000, 15000, 6000, 500)

        user_row = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Age": age,
                    "Occupation": occupation,
                    "Sleep Duration": sleep_duration,
                    "Physical Activity Level": activity,
                    "BMI Category": bmi,
                    "Systolic BP": systolic,
                    "Diastolic BP": diastolic,
                    "Heart Rate": heart_rate,
                    "Daily Steps": steps,
                }
            ]
        )

        if st.button("Predict", type="primary"):
            stress_fitted, _, _ = train_regressors(lifestyle, "Stress Level", tuple(
                c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Stress Level"]
            ))
            quality_fitted, _, _ = train_regressors(lifestyle, "Quality of Sleep", tuple(
                c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Quality of Sleep"]
            ))
            disorder_fitted, _, _, disorder_classes = train_classifiers(lifestyle, "Sleep Disorder", tuple(
                c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Sleep Disorder"]
            ))

            stress_pred = stress_fitted["Random Forest"].predict(user_row)[0]
            quality_pred = quality_fitted["Random Forest"].predict(user_row)[0]
            disorder_model = disorder_fitted["Random Forest"]
            disorder_proba = disorder_model.predict_proba(user_row)[0]

            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Stress Level (0-10)", f"{stress_pred:.1f}")
            r2.metric("Predicted Sleep Quality (1-10)", f"{quality_pred:.1f}")
            top_class_idx = int(np.argmax(disorder_proba))
            r3.metric("Most likely Sleep Disorder status", disorder_model.classes_[top_class_idx])

            st.write("Sleep disorder probability breakdown:")
            proba_df = pd.DataFrame(
                {"Status": disorder_model.classes_, "Probability": disorder_proba}
            ).sort_values("Probability", ascending=False)
            st.dataframe(proba_df, use_container_width=True, hide_index=True)

    else:
        efficiency = load_efficiency()
        base = efficiency[EFFICIENCY_PREDICTOR_FEATURES]

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 9, 90, 30, key="eff_age")
            gender = st.selectbox("Gender", sorted(base["Gender"].dropna().unique()), key="eff_gender")
            sleep_duration = st.slider("Sleep duration (hours)", 3.0, 12.0, 7.5, 0.1, key="eff_duration")
        with col2:
            caffeine = st.slider("Caffeine consumption (mg)", 0, 200, 0, 25)
            alcohol = st.slider("Alcohol consumption (units)", 0, 5, 0)
            smoking = st.selectbox("Smoking status", sorted(base["Smoking status"].dropna().unique()))
            exercise = st.slider("Exercise frequency (times/week)", 0, 7, 3)

        user_row = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Sleep duration": sleep_duration,
                    "Caffeine consumption": caffeine,
                    "Alcohol consumption": alcohol,
                    "Smoking status": smoking,
                    "Exercise frequency": exercise,
                }
            ]
        )

        if st.button("Predict", type="primary"):
            timestamp_cols = ("ID", "Bedtime", "Wakeup time")
            extra_drop = tuple(
                c for c in efficiency.columns if c not in [*EFFICIENCY_PREDICTOR_FEATURES, "Sleep efficiency"]
            )
            eff_fitted, _, _ = train_regressors(efficiency, "Sleep efficiency", timestamp_cols + extra_drop)
            extra_drop_awake = tuple(
                c for c in efficiency.columns if c not in [*EFFICIENCY_PREDICTOR_FEATURES, "Awakenings"]
            )
            awake_fitted, _, _ = train_regressors(efficiency, "Awakenings", timestamp_cols + extra_drop_awake)

            eff_pred = eff_fitted["Random Forest"].predict(user_row)[0]
            awake_pred = awake_fitted["Random Forest"].predict(user_row)[0]

            r1, r2 = st.columns(2)
            r1.metric("Predicted Sleep Efficiency", f"{eff_pred:.0%}")
            r2.metric("Predicted Awakenings per night", f"{awake_pred:.1f}")


# ---------------------------------------------------------------------------
# Page: Full report
# ---------------------------------------------------------------------------
def page_report():
    st.title("Full Written Report")
    report_path = ROOT / "Model_Report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.error("Model_Report.md was not found in the project folder.")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "Overview": page_overview,
    "Explore the Data": page_explore,
    "Model Performance": page_model_performance,
    "Try the Predictor": page_predictor,
    "Full Report": page_report,
}


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Student biomedical research project · Public Kaggle sleep datasets · "
        "Educational use only, not medical advice."
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
