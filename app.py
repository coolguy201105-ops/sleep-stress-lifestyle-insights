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

# Brand color palette used everywhere below (cards, gauges, charts, badges).
BLUE = "#2563EB"
GREEN = "#16A34A"
AMBER = "#F59E0B"
RED = "#DC2626"
VIOLET = "#7C3AED"
TEAL = "#0D9488"
GRAY = "#6B7280"

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


def feature_importance_chart(pipe: Pipeline, feature_names: list[str], top_n: int = 8, color: str = VIOLET):
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
    fade = np.linspace(1.0, 0.35, len(importance))
    bar_colors = [_hex_fade(color, f) for f in fade]
    ax.barh(importance.index[::-1], importance.values[::-1], color=bar_colors[::-1])
    ax.set_xlabel("Importance")
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def _hex_fade(hex_color: str, strength: float) -> str:
    """Blend a hex color toward white based on strength (1.0 = full color)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r * strength + 255 * (1 - strength))
    g = int(g * strength + 255 * (1 - strength))
    b = int(b * strength + 255 * (1 - strength))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Colorful UI building blocks
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .stat-card {
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 10px;
            color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .stat-card .label {
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            opacity: 0.9;
        }
        .stat-card .value {
            font-size: 30px;
            font-weight: 800;
            line-height: 1.2;
            margin-top: 2px;
        }
        .stat-card .sub {
            font-size: 13px;
            opacity: 0.92;
            margin-top: 4px;
        }
        .gauge-wrap { margin-bottom: 18px; }
        .gauge-label {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 6px;
            color: #1F2937;
        }
        .gauge-track {
            background: #E5E7EB;
            border-radius: 10px;
            height: 16px;
            overflow: hidden;
        }
        .gauge-fill { height: 100%; border-radius: 10px; }
        .section-band {
            border-radius: 10px;
            padding: 10px 16px;
            font-weight: 700;
            font-size: 15px;
            margin: 6px 0 14px 0;
            color: white;
        }
        .takeaway {
            border-left: 6px solid var(--tk-color, #16A34A);
            background: color-mix(in srgb, var(--tk-color, #16A34A) 10%, white);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            margin: 10px 0 18px 0;
        }
        [data-testid="stSidebar"] {
            min-width: 320px;
        }
        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            padding: 16px 18px;
            margin-bottom: 10px;
            font-size: 17px;
            font-weight: 600;
            border-radius: 12px;
            min-height: 56px;
            line-height: 1.3;
        }
        [data-testid="stSidebar"] .stButton button p {
            text-align: left;
            font-size: 17px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, color: str, sub: str = ""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stat-card" style="background:{color};">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_bar(label: str, pct: float, color: str, right_text: str | None = None):
    right_text = right_text if right_text is not None else f"{pct:.0f}%"
    pct_clamped = max(2, min(100, pct))
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge-label"><span>{label}</span><span style="color:{color};">{right_text}</span></div>
            <div class="gauge-track"><div class="gauge-fill" style="width:{pct_clamped}%; background:{color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_band(text: str, color: str):
    st.markdown(f'<div class="section-band" style="background:{color};">{text}</div>', unsafe_allow_html=True)


def takeaway(text: str, color: str):
    st.markdown(
        f'<div class="takeaway" style="--tk-color:{color};">{text}</div>',
        unsafe_allow_html=True,
    )


def go_to(page_key: str):
    st.session_state["nav_choice"] = page_key
    st.rerun()


def r2_color(r2: float) -> tuple[str, str]:
    if r2 >= 0.8:
        return GREEN, "Excellent fit"
    if r2 >= 0.5:
        return BLUE, "Moderate fit"
    if r2 >= 0.2:
        return AMBER, "Weak fit"
    return RED, "Little power"


def accuracy_color(accuracy: float, dummy_accuracy: float) -> tuple[str, str]:
    gain = accuracy - dummy_accuracy
    if gain >= 0.25:
        return GREEN, "Strong gain"
    if gain >= 0.10:
        return BLUE, "Moderate gain"
    return AMBER, "Small gain"


def footer():
    st.markdown("---")
    st.caption("🎓 Student project · Public Kaggle data · Educational use only, not medical advice.")


# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
def page_overview():
    st.title("🌙 Sleep, Stress & Lifestyle Insights")
    st.caption("Evidence-based takeaways on sleep and stress — for students and families in Irvine, CA")

    c1, c2, c3 = st.columns(3)
    with c1:
        stat_card("Dataset 1", "132 people", BLUE, "Sleep, stress & lifestyle")
    with c2:
        stat_card("Dataset 2", "452 people", TEAL, "Sleep efficiency & habits")
    with c3:
        stat_card("Models compared", "3 per outcome", VIOLET, "Baseline → Regression → Forest")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📊 Explore the Data", use_container_width=True):
            go_to("📊 Explore the Data")
    with b2:
        if st.button("🔮 Try the Predictor", use_container_width=True, type="primary"):
            go_to("🔮 Try the Predictor")
    with b3:
        if st.button("🤖 See How Accurate", use_container_width=True):
            go_to("🤖 Model Performance")

    st.info(
        "Public, de-identified data — not Irvine-specific. Results show *patterns*, not proof of cause and effect.",
        icon="💡",
    )
    footer()


# ---------------------------------------------------------------------------
# Page: Explore the data
# ---------------------------------------------------------------------------
def page_explore():
    st.title("📊 Explore the Data")
    lifestyle = load_lifestyle()
    efficiency = load_efficiency()

    dataset_choice = st.radio(
        "Dataset", ["😴 Sleep Health & Lifestyle", "⏱️ Sleep Efficiency"], horizontal=True, label_visibility="collapsed"
    )

    if dataset_choice.startswith("😴"):
        df = lifestyle
        section_band("Less sleep → more stress?", BLUE)
        plot_data = df.dropna(subset=["Sleep Duration", "Stress Level"]).copy()
        plot_data["Sleep category"] = pd.cut(
            plot_data["Sleep Duration"],
            bins=[0, 7, 8, float("inf")],
            labels=["Less than 7h", "7-8h", "More than 8h"],
            right=False,
        )
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(
            data=plot_data, x="Sleep category", y="Stress Level", errorbar=None, ax=ax,
            palette=[RED, AMBER, GREEN], hue="Sleep category", legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Avg. stress (0-10)")
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        takeaway("Under 7 hours of sleep lines up with noticeably higher stress.", RED)

        section_band("Sleep quality vs. stress & sleep disorders", VIOLET)
        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        sns.scatterplot(
            data=df, x="Quality of Sleep", y="Stress Level", hue="Sleep Disorder", ax=ax2,
            palette={"None": GREEN, "Insomnia": AMBER, "Sleep Apnea": RED}, s=60,
        )
        ax2.set_xlabel("Sleep quality (1-10)")
        ax2.set_ylabel("Stress (0-10)")
        ax2.set_facecolor("#FAFAFA")
        fig2.patch.set_facecolor("#FAFAFA")
        sns.despine(ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
        takeaway("Lower quality sleep clusters with higher stress and more sleep disorders.", VIOLET)
    else:
        df = efficiency
        section_band("Longer sleep ≠ better sleep", TEAL)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.scatterplot(
            data=df, x="Sleep duration", y="Sleep efficiency", hue="Smoking status", ax=ax,
            palette={"Yes": RED, "No": GREEN}, s=60,
        )
        ax.set_xlabel("Sleep duration (hours)")
        ax.set_ylabel("Sleep efficiency")
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FAFAFA")
        sns.despine(ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        takeaway("Sleeping longer doesn't guarantee more efficient sleep.", TEAL)

        section_band("Does caffeine hurt sleep efficiency?", AMBER)
        fig2, ax2 = plt.subplots(figsize=(7, 3.8))
        sns.scatterplot(data=df, x="Caffeine consumption", y="Sleep efficiency", ax=ax2, color=AMBER, s=60)
        ax2.set_xlabel("Caffeine (mg)")
        ax2.set_ylabel("Sleep efficiency")
        ax2.set_facecolor("#FAFAFA")
        fig2.patch.set_facecolor("#FAFAFA")
        sns.despine(ax=ax2)
        fig2.tight_layout()
        st.pyplot(fig2)
        takeaway("Higher caffeine intake tracks with lower, more scattered sleep efficiency.", AMBER)

    with st.expander("🔍 See the raw data table & summary stats"):
        st.dataframe(df.head(20), use_container_width=True)
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    footer()


# ---------------------------------------------------------------------------
# Page: Model performance
# ---------------------------------------------------------------------------
def metric_legend():
    cols = st.columns(4)
    legend = [
        ("MAE / RMSE", "Lower = better", GRAY),
        ("R²", "Higher = better", GREEN),
        ("Accuracy", "% correct", BLUE),
        ("Dummy baseline", "The bar to beat", AMBER),
    ]
    for col, (label, sub, color) in zip(cols, legend):
        with col:
            st.markdown(
                f'<span style="background:{color}22; color:{color}; padding:3px 10px; '
                f'border-radius:999px; font-size:12px; font-weight:700;">{label}</span>',
                unsafe_allow_html=True,
            )
            st.caption(sub)


def _regression_tab(df, target, drop_columns, unit_hint, question, color):
    fitted, metrics, features = train_regressors(df, target, drop_columns)
    dummy_row = metrics[metrics["Model"] == "Dummy baseline"].iloc[0]
    best_row = metrics.sort_values("R2", ascending=False).iloc[0]

    section_band(question, color)
    fit_color, verdict = r2_color(best_row["R2"])
    gauge_bar(f"{best_row['Model']} — % of pattern explained (R²)", best_row["R2"] * 100, fit_color, f"{best_row['R2']:.0%} · {verdict}")
    st.caption(f"Typical miss: {best_row['MAE']:.2f} {unit_hint} (vs. {dummy_row['MAE']:.2f} if just guessing the average).")

    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    fig = feature_importance_chart(fitted["Random Forest"], features, color=color)
    if fig:
        st.caption("🔑 Top drivers of this prediction")
        st.pyplot(fig)


def _classification_tab(df, target, drop_columns, question, color):
    fitted, metrics, features, classes = train_classifiers(df, target, drop_columns)
    dummy_row = metrics[metrics["Model"] == "Dummy baseline"].iloc[0]
    best_row = metrics.sort_values("Balanced Accuracy", ascending=False).iloc[0]

    section_band(question, color)
    acc_color, verdict = accuracy_color(best_row["Accuracy"], dummy_row["Accuracy"])
    gauge_bar(f"{best_row['Model']} — accuracy", best_row["Accuracy"] * 100, acc_color, f"{best_row['Accuracy']:.0%} · {verdict}")
    st.caption(f"Categories: {', '.join(classes)}")

    st.dataframe(metrics.round(3), use_container_width=True, hide_index=True)
    fig = feature_importance_chart(fitted["Random Forest"], features, color=color)
    if fig:
        st.caption("🔑 Top drivers of this prediction")
        st.pyplot(fig)


def page_model_performance():
    st.title("🤖 Model Performance")
    metric_legend()
    st.markdown("")

    lifestyle = load_lifestyle()
    efficiency = load_efficiency()
    timestamp_cols = ("ID", "Bedtime", "Wakeup time")

    tabs = st.tabs(["😣 Stress", "😴 Sleep Quality", "🩺 Sleep Disorder", "⏱️ Efficiency", "🌙 Awakenings"])

    with tabs[0]:
        _regression_tab(lifestyle, "Stress Level", (), "points (0-10)", "Can we predict stress level from sleep & lifestyle?", BLUE)
    with tabs[1]:
        _regression_tab(lifestyle, "Quality of Sleep", (), "points (1-10)", "Can we predict self-rated sleep quality?", VIOLET)
    with tabs[2]:
        _classification_tab(lifestyle, "Sleep Disorder", (), "Can we tell apart no disorder, insomnia, and sleep apnea?", RED)
    with tabs[3]:
        _regression_tab(efficiency, "Sleep efficiency", timestamp_cols, "points of efficiency", "Can we predict sleep efficiency?", TEAL)
    with tabs[4]:
        _regression_tab(efficiency, "Awakenings", timestamp_cols, "awakenings/night", "Can we predict nightly awakenings?", AMBER)

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
            age = st.slider("Age", 18, 80, 25)
            occupation_options = sorted(base["Occupation"].dropna().unique().tolist()) + ["Unemployed", "Retired"]
            occupation = st.selectbox(
                "Occupation", occupation_options,
                help="Pick whichever is closest to your situation, or choose Unemployed/Retired.",
            )
        with col2:
            sleep_duration = st.slider("Sleep duration (hrs/night)", 3.0, 10.0, 7.0, 0.1)
            activity = st.slider("Activity level (0-100)", 0, 100, 50)
            bmi = st.selectbox(
                "Body weight category", sorted(base["BMI Category"].dropna().unique()),
                help="If unsure, pick 'Normal' as a default.",
            )
        with col3:
            systolic = st.slider("Blood pressure — top (systolic)", 90, 180, 120, help="Normal is roughly 120.")
            diastolic = st.slider("Blood pressure — bottom (diastolic)", 60, 120, 80, help="Normal is roughly 80.")
            heart_rate = st.slider("Resting heart rate (bpm)", 50, 110, 72)
            steps = st.slider("Daily steps", 1000, 15000, 6000, 500)

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

    st.markdown("### Your results")
    stress_color = GREEN if stress_pred <= 3.5 else AMBER if stress_pred <= 6.5 else RED
    quality_color = RED if quality_pred <= 3.5 else AMBER if quality_pred <= 6.5 else GREEN

    c1, c2 = st.columns(2)
    with c1:
        gauge_bar("Predicted stress level", stress_pred * 10, stress_color, f"{stress_pred:.1f} / 10")
    with c2:
        gauge_bar("Predicted sleep quality", quality_pred * 10, quality_color, f"{quality_pred:.1f} / 10")

    class_colors = {"None": GREEN, "Insomnia": AMBER, "Sleep Apnea": RED}
    order = np.argsort(-disorder_proba)
    st.write("**Sleep-disorder likelihood:**")
    for idx in order:
        cls = disorder_model.classes_[idx]
        gauge_bar(cls, disorder_proba[idx] * 100, class_colors.get(cls, GRAY))

    if stress_pred > 6.5:
        st.error("Predicted stress is high — consider more sleep, more activity, or talking to someone you trust.")
    elif quality_pred <= 3.5:
        st.warning("Predicted sleep quality is low — small routine changes may help.")
    else:
        st.success("These habits look solid based on the data!")


def _efficiency_predictor():
    efficiency = load_efficiency()
    base = efficiency[EFFICIENCY_PREDICTOR_FEATURES]

    with st.form("efficiency_predictor_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 9, 90, 30, key="eff_age")
            gender = st.selectbox("Gender", sorted(base["Gender"].dropna().unique()), key="eff_gender")
            sleep_duration = st.slider("Sleep duration (hrs/night)", 3.0, 12.0, 7.5, 0.1, key="eff_duration")
        with col2:
            caffeine = st.slider("Caffeine (mg/day)", 0, 200, 0, 25, help="A cup of coffee is roughly 95mg.")
            alcohol = st.slider("Alcoholic drinks/day", 0, 5, 0)
            smoking = st.selectbox("Do you smoke?", sorted(base["Smoking status"].dropna().unique()))
            exercise = st.slider("Exercise (times/week)", 0, 7, 3)

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
    eff_color = GREEN if eff_pred >= 0.85 else AMBER if eff_pred >= 0.7 else RED
    awake_color = GREEN if awake_pred <= 1 else AMBER if awake_pred <= 3 else RED

    c1, c2 = st.columns(2)
    with c1:
        gauge_bar("Predicted sleep efficiency", eff_pred * 100, eff_color, f"{eff_pred:.0%}")
    with c2:
        gauge_bar("Predicted awakenings/night", min(awake_pred, 10) * 10, awake_color, f"{awake_pred:.1f}")

    if eff_pred < 0.7:
        st.error("Predicted efficiency is low — caffeine, alcohol, and irregular schedules are common culprits.")
    else:
        st.success("Predicted sleep efficiency looks solid!")


def page_predictor():
    st.title("🔮 Try the Predictor")
    st.warning("Educational demo only — not a medical tool or diagnosis.", icon="🩺")

    choice = st.radio(
        "Predict:",
        ["😣 Stress, quality & disorder risk", "⏱️ Efficiency & awakenings"],
        horizontal=True,
        label_visibility="collapsed",
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
    inject_css()
    if "nav_choice" not in st.session_state:
        st.session_state["nav_choice"] = "🏠 Overview"

    with st.sidebar:
        st.markdown("## 🌙 Sleep Insights")
        st.markdown("")
        for page_key in PAGES:
            is_active = st.session_state["nav_choice"] == page_key
            if st.button(
                page_key,
                key=f"nav_btn_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["nav_choice"] = page_key
                st.rerun()
        st.markdown("---")
        st.caption("🎓 Student project\n\n📦 Public Kaggle data\n\n⚠️ Not medical advice")

    PAGES[st.session_state["nav_choice"]]()


if __name__ == "__main__":
    main()
