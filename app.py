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


@st.cache_resource(show_spinner="Crunching the numbers...")
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


@st.cache_resource(show_spinner="Crunching the numbers...")
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


def feature_importance_chart(pipe: Pipeline, feature_names: list[str], top_n: int = 8):
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

    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=importance.values, y=importance.index, ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Small UI helpers to keep things friendly and consistent
# ---------------------------------------------------------------------------
def go_to(page_key: str):
    st.session_state["nav_choice"] = page_key
    st.rerun()


def metric_glossary():
    with st.expander("📖 New to this? Tap here for a plain-English explanation of these numbers"):
        st.markdown(
            """
            - **MAE (average error)** — on average, how far off a prediction is, in real units
              (e.g., "off by 0.2 points"). Lower is better.
            - **RMSE** — similar to MAE, but penalizes big mistakes more heavily. Lower is better.
            - **R² (percent explained)** — what percent of the pattern in the data the model
              successfully captures, from 0% to 100%. Higher is better.
            - **Accuracy** — the percent of predictions the model got exactly right.
            - **Balanced Accuracy** — accuracy that's been adjusted so a common category (like
              "no sleep disorder") can't make the score look artificially good.
            - **Macro F1** — a fairness-balanced score across every category, from 0 to 1.
            - **Dummy baseline** — a "dumb" model that always guesses the average or the most
              common answer. It's the bar every real model needs to beat.
            """
        )


def r2_badge(r2: float) -> tuple[str, str]:
    if r2 >= 0.8:
        return "success", f"🟢 Excellent fit — explains about {r2:.0%} of the pattern in the data."
    if r2 >= 0.5:
        return "info", f"🟡 Moderate fit — explains about {r2:.0%} of the pattern in the data."
    if r2 >= 0.2:
        return "warning", f"🟠 Weak fit — only explains about {r2:.0%} of the pattern in the data."
    return "error", f"🔴 Little to no predictive power (explains about {r2:.0%})."


def accuracy_badge(accuracy: float, dummy_accuracy: float) -> tuple[str, str]:
    gain = accuracy - dummy_accuracy
    if gain >= 0.25:
        return "success", f"🟢 Strong improvement — {accuracy:.0%} correct, {gain:.0%} points better than guessing."
    if gain >= 0.1:
        return "info", f"🟡 Moderate improvement — {accuracy:.0%} correct, {gain:.0%} points better than guessing."
    return "warning", f"🟠 Only a small improvement over guessing ({accuracy:.0%} correct)."


def show_badge(kind: str, text: str):
    getattr(st, kind)(text)


def footer():
    st.markdown("---")
    st.caption(
        "🎓 Student biomedical research project · Public Kaggle sleep datasets · "
        "Educational use only — not medical advice."
    )


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("🌙 Sleep, Stress & Lifestyle Insights")
    st.subheader("Simple, evidence-based takeaways about sleep and stress — built for students and families in Irvine, CA")

    st.markdown(
        """
        We looked at two public sleep-health datasets to answer a simple question:
        **what lifestyle habits are most connected to better sleep and lower stress?**
        This site lets you explore what we found, see how well our predictions actually
        work, and even try the models yourself.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Explore the Data", use_container_width=True):
            go_to("📊 Explore the Data")
    with c2:
        if st.button("🔮 Try the Predictor", use_container_width=True):
            go_to("🔮 Try the Predictor")
    with c3:
        if st.button("🤖 See How Accurate It Is", use_container_width=True):
            go_to("🤖 Model Performance")

    st.markdown("### What data did we use?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 😴 Sleep Health & Lifestyle")
        st.write(
            "132 people (after cleaning). Tracks sleep duration/quality, stress, "
            "activity, BMI, blood pressure, heart rate, steps, and sleep disorders."
        )
    with col2:
        st.markdown("#### ⏱️ Sleep Efficiency")
        st.write(
            "452 people. Tracks sleep duration/efficiency, sleep stages, "
            "night-time awakenings, caffeine/alcohol use, and exercise."
        )

    st.info(
        "**Good to know:** this data is public and not specific to Irvine, and it shows "
        "*patterns*, not proof that one thing *causes* another. Think of this as a "
        "starting point for healthier habits, not a diagnosis.",
        icon="💡",
    )
    footer()


# ---------------------------------------------------------------------------
# Page: Explore the data
# ---------------------------------------------------------------------------
def page_explore():
    st.title("📊 Explore the Data")
    st.caption("A quick visual tour of what's in each dataset — no numbers background needed.")
    lifestyle = load_lifestyle()
    efficiency = load_efficiency()

    dataset_choice = st.radio(
        "Pick a dataset to look at:",
        ["😴 Sleep Health & Lifestyle", "⏱️ Sleep Efficiency"],
        horizontal=True,
    )

    if dataset_choice.startswith("😴"):
        df = lifestyle
        st.markdown("#### Do people who sleep less report more stress?")
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
        ax.set_ylabel("Average stress level (0-10)")
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        st.success(
            "👉 **Takeaway:** people sleeping less than 7 hours report noticeably higher "
            "average stress than those sleeping 8+ hours in this data."
        )

        st.markdown("#### Does sleep quality relate to stress and sleep disorders?")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Quality of Sleep", y="Stress Level", hue="Sleep Disorder", ax=ax2)
        ax2.set_xlabel("Sleep quality (1-10)")
        ax2.set_ylabel("Stress level (0-10)")
        sns.despine(ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.success("👉 **Takeaway:** lower sleep quality tends to line up with higher stress, and people with a sleep disorder cluster toward the lower-quality, higher-stress corner.")
    else:
        df = efficiency
        st.markdown("#### Does sleeping longer mean sleeping *better*?")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Sleep duration", y="Sleep efficiency", hue="Smoking status", ax=ax)
        ax.set_xlabel("Sleep duration (hours)")
        ax.set_ylabel("Sleep efficiency (share of time in bed actually asleep)")
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        st.success("👉 **Takeaway:** longer sleep doesn't automatically mean more efficient sleep — quality and duration aren't the same thing.")

        st.markdown("#### Does caffeine hurt sleep efficiency?")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="Caffeine consumption", y="Sleep efficiency", ax=ax2)
        ax2.set_xlabel("Caffeine consumption (mg)")
        ax2.set_ylabel("Sleep efficiency")
        sns.despine(ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.success("👉 **Takeaway:** higher caffeine intake is associated with more scattered, generally lower sleep efficiency.")

    with st.expander("🔍 Curious about the raw numbers? See the data table and summary statistics"):
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    footer()


# ---------------------------------------------------------------------------
# Page: Model performance
# ---------------------------------------------------------------------------
def _regression_tab(df, target, drop_columns, unit_hint, plain_name):
    fitted, metrics, features = train_regressors(df, target, drop_columns)
    dummy_row = metrics[metrics["Model"] == "Dummy baseline"].iloc[0]
    best_row = metrics.sort_values("R2", ascending=False).iloc[0]

    st.markdown(f"**In plain terms:** {plain_name}")
    kind, text = r2_badge(best_row["R2"])
    show_badge(kind, f"Best model ({best_row['Model']}): {text}")
    st.caption(f"Typical prediction is off by about {best_row['MAE']:.2f} {unit_hint} — compared to {dummy_row['MAE']:.2f} {unit_hint} if we just guessed the average.")

    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    fig = feature_importance_chart(fitted["Random Forest"], features)
    if fig:
        st.caption("🔑 What mattered most to the model's predictions:")
        st.pyplot(fig)


def _classification_tab(df, target, drop_columns, plain_name):
    fitted, metrics, features, classes = train_classifiers(df, target, drop_columns)
    dummy_row = metrics[metrics["Model"] == "Dummy baseline"].iloc[0]
    best_row = metrics.sort_values("Balanced Accuracy", ascending=False).iloc[0]

    st.markdown(f"**In plain terms:** {plain_name}")
    kind, text = accuracy_badge(best_row["Accuracy"], dummy_row["Accuracy"])
    show_badge(kind, f"Best model ({best_row['Model']}): {text}")
    st.caption(f"Possible categories: {', '.join(classes)}.")

    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    fig = feature_importance_chart(fitted["Random Forest"], features)
    if fig:
        st.caption("🔑 What mattered most to the model's predictions:")
        st.pyplot(fig)


def page_model_performance():
    st.title("🤖 Model Performance")
    st.write(
        "For every outcome below, we trained 3 models on the same people and tested them on "
        "data they'd never seen: a **dummy guess**, a **simple regression**, and a more "
        "powerful **random forest**. Better models should beat the dummy guess by a lot."
    )
    metric_glossary()

    lifestyle = load_lifestyle()
    efficiency = load_efficiency()
    timestamp_cols = ("ID", "Bedtime", "Wakeup time")

    tabs = st.tabs(["😣 Stress Level", "😴 Sleep Quality", "🩺 Sleep Disorder", "⏱️ Sleep Efficiency", "🌙 Awakenings"])

    with tabs[0]:
        _regression_tab(
            lifestyle, "Stress Level", (), "points (0-10 scale)",
            "how well can we predict someone's stress level (0-10) from their sleep and lifestyle info?",
        )
    with tabs[1]:
        _regression_tab(
            lifestyle, "Quality of Sleep", (), "points (1-10 scale)",
            "how well can we predict someone's self-rated sleep quality (1-10)?",
        )
    with tabs[2]:
        _classification_tab(
            lifestyle, "Sleep Disorder", (),
            "can we tell whether someone likely has no sleep disorder, insomnia, or sleep apnea?",
        )
    with tabs[3]:
        _regression_tab(
            efficiency, "Sleep efficiency", timestamp_cols, "percentage points of efficiency",
            "how well can we predict what share of time in bed someone actually spends asleep?",
        )
    with tabs[4]:
        _regression_tab(
            efficiency, "Awakenings", timestamp_cols, "awakenings per night",
            "how well can we predict how many times someone wakes up during the night?",
        )

    footer()


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


def _stress_quality_disorder_predictor():
    lifestyle = load_lifestyle()
    base = lifestyle[LIFESTYLE_PREDICTOR_FEATURES]

    with st.form("lifestyle_predictor_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender", sorted(base["Gender"].dropna().unique()))
            age = st.slider("Age", 18, 80, 25, help="Your age in years.")
            occupation = st.selectbox(
                "Occupation", sorted(base["Occupation"].dropna().unique()),
                help="Pick whichever is closest to your situation (students often relate most to office/desk-based roles).",
            )
        with col2:
            sleep_duration = st.slider(
                "Sleep duration (hours/night)", 3.0, 10.0, 7.0, 0.1,
                help="About how many hours you sleep on a typical night.",
            )
            activity = st.slider(
                "Physical activity (0 = none, 100 = very active)", 0, 100, 50,
                help="A rough score for how physically active you are day to day.",
            )
            bmi = st.selectbox(
                "Body weight category", sorted(base["BMI Category"].dropna().unique()),
                help="Body Mass Index category, if known. Skip guessing if unsure — pick 'Normal' as a default.",
            )
        with col3:
            systolic = st.slider("Blood pressure — top number (systolic)", 90, 180, 120, help="A normal reading is roughly 120.")
            diastolic = st.slider("Blood pressure — bottom number (diastolic)", 60, 120, 80, help="A normal reading is roughly 80.")
            heart_rate = st.slider("Resting heart rate (beats per minute)", 50, 110, 72)
            steps = st.slider("Daily steps", 1000, 15000, 6000, 500, help="Roughly how many steps you walk per day.")

        submitted = st.form_submit_button("🔮 Get my prediction", type="primary", use_container_width=True)

    if not submitted:
        return

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

    with st.spinner("Thinking..."):
        stress_fitted, _, _ = train_regressors(
            lifestyle, "Stress Level",
            tuple(c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Stress Level"]),
        )
        quality_fitted, _, _ = train_regressors(
            lifestyle, "Quality of Sleep",
            tuple(c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Quality of Sleep"]),
        )
        disorder_fitted, _, _, _ = train_classifiers(
            lifestyle, "Sleep Disorder",
            tuple(c for c in lifestyle.columns if c not in [*LIFESTYLE_PREDICTOR_FEATURES, "Sleep Disorder"]),
        )

    stress_pred = float(stress_fitted["Random Forest"].predict(user_row)[0])
    quality_pred = float(quality_fitted["Random Forest"].predict(user_row)[0])
    disorder_model = disorder_fitted["Random Forest"]
    disorder_proba = disorder_model.predict_proba(user_row)[0]
    top_class_idx = int(np.argmax(disorder_proba))

    st.markdown("### Your results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted stress level", f"{stress_pred:.1f} / 10")
    r2.metric("Predicted sleep quality", f"{quality_pred:.1f} / 10")
    r3.metric("Most likely sleep-disorder status", disorder_model.classes_[top_class_idx])

    if stress_pred <= 3.5:
        st.success("🟢 Predicted stress looks low based on these habits.")
    elif stress_pred <= 6.5:
        st.warning("🟡 Predicted stress is moderate — a few habit tweaks could help.")
    else:
        st.error("🔴 Predicted stress looks high based on these habits. Consider more sleep, more activity, or talking to someone you trust.")

    st.write("**Sleep-disorder likelihood breakdown:**")
    proba_df = pd.DataFrame(
        {"Status": disorder_model.classes_, "Probability": disorder_proba}
    ).sort_values("Probability", ascending=False)
    proba_df["Probability"] = proba_df["Probability"].map(lambda p: f"{p:.0%}")
    st.dataframe(proba_df, use_container_width=True, hide_index=True)


def _efficiency_predictor():
    efficiency = load_efficiency()
    base = efficiency[EFFICIENCY_PREDICTOR_FEATURES]

    with st.form("efficiency_predictor_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 9, 90, 30, key="eff_age")
            gender = st.selectbox("Gender", sorted(base["Gender"].dropna().unique()), key="eff_gender")
            sleep_duration = st.slider("Sleep duration (hours/night)", 3.0, 12.0, 7.5, 0.1, key="eff_duration")
        with col2:
            caffeine = st.slider("Caffeine (mg/day)", 0, 200, 0, 25, help="A cup of coffee is roughly 95mg.")
            alcohol = st.slider("Alcoholic drinks (per day)", 0, 5, 0)
            smoking = st.selectbox("Do you smoke?", sorted(base["Smoking status"].dropna().unique()))
            exercise = st.slider("Exercise (times per week)", 0, 7, 3)

        submitted = st.form_submit_button("🔮 Get my prediction", type="primary", use_container_width=True)

    if not submitted:
        return

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

    with st.spinner("Thinking..."):
        timestamp_cols = ("ID", "Bedtime", "Wakeup time")
        extra_drop = tuple(c for c in efficiency.columns if c not in [*EFFICIENCY_PREDICTOR_FEATURES, "Sleep efficiency"])
        eff_fitted, _, _ = train_regressors(efficiency, "Sleep efficiency", timestamp_cols + extra_drop)
        extra_drop_awake = tuple(c for c in efficiency.columns if c not in [*EFFICIENCY_PREDICTOR_FEATURES, "Awakenings"])
        awake_fitted, _, _ = train_regressors(efficiency, "Awakenings", timestamp_cols + extra_drop_awake)

    eff_pred = float(eff_fitted["Random Forest"].predict(user_row)[0])
    awake_pred = float(awake_fitted["Random Forest"].predict(user_row)[0])

    st.markdown("### Your results")
    r1, r2 = st.columns(2)
    r1.metric("Predicted sleep efficiency", f"{eff_pred:.0%}")
    r2.metric("Predicted awakenings per night", f"{awake_pred:.1f}")

    if eff_pred >= 0.85:
        st.success("🟢 Predicted sleep efficiency looks great!")
    elif eff_pred >= 0.7:
        st.warning("🟡 Predicted sleep efficiency is okay, but there's room to improve.")
    else:
        st.error("🔴 Predicted sleep efficiency looks low — caffeine, alcohol, and irregular schedules are common culprits.")


def page_predictor():
    st.title("🔮 Try the Predictor")
    st.warning(
        "This is an educational demo based on public survey data, not a medical tool. "
        "It does not diagnose conditions or replace advice from a doctor.",
        icon="🩺",
    )
    st.write("Fill in the form below with real or hypothetical info, then hit predict.")

    choice = st.radio(
        "What would you like to predict?",
        ["😣 Stress, sleep quality & sleep-disorder risk", "⏱️ Sleep efficiency & awakenings"],
    )

    if choice.startswith("😣"):
        _stress_quality_disorder_predictor()
    else:
        _efficiency_predictor()

    footer()


# ---------------------------------------------------------------------------
# Page: Full report
# ---------------------------------------------------------------------------
def page_report():
    st.title("📄 Full Written Report")
    st.caption("The complete methodology and results, for anyone who wants the full details.")
    report_path = ROOT / "Model_Report.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.error("Model_Report.md was not found in the project folder.")
    footer()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Overview": page_overview,
    "📊 Explore the Data": page_explore,
    "🤖 Model Performance": page_model_performance,
    "🔮 Try the Predictor": page_predictor,
    "📄 Full Report": page_report,
}


def main():
    with st.sidebar:
        st.markdown("## 🌙 Sleep Insights")
        choice = st.radio("Go to:", list(PAGES.keys()), key="nav_choice", label_visibility="collapsed")
        st.markdown("---")
        st.caption(
            "🎓 Student biomedical research project\n\n"
            "📦 Public Kaggle sleep datasets\n\n"
            "⚠️ Educational use only, not medical advice"
        )
    PAGES[choice]()


if __name__ == "__main__":
    main()
