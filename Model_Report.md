# Model Report: Sleep, Stress, and Lifestyle Baseline Models

This report documents the models trained in `baseline_models.py`, run against the two human
datasets (`Sleep_health_and_lifestyle_dataset.csv` and `Sleep_Efficiency.csv`). For every
biomedical target variable, three models of increasing complexity were trained and compared
on the same held-out test rows (25% of the data, never seen during training):

1. **Dummy baseline** — a naive reference point.
2. **Linear / Logistic Regression** — a simple, fully interpretable model.
3. **Random Forest** — a flexible model that can capture non-linear patterns and interactions.

Comparing all three on identical test data shows whether the added complexity of Random
Forest is actually earning its keep, or whether a much simpler model explains the data just
as well.

---

## 1. Dummy Baseline (`DummyRegressor` / `DummyClassifier`)

**How it works:** This model ignores every input feature. For regression targets, it always
predicts the training-set mean (e.g., "predict everyone has the average stress level"). For
the classification target, it always predicts the single most common class (e.g., "predict
everyone has no sleep disorder," since that is the majority category).

**Purpose in this project:** It is not meant to be useful on its own — it exists purely as a
sanity-check floor. If a "smart" model cannot beat this dumb guess, that model has learned
nothing real from the data.

**What we can do with it:** Use it as the yardstick in every results table. Any claim like
"our model predicts stress level well" only means something when it is stated relative to
this baseline (e.g., "cut prediction error by 84% versus guessing the average").

---

## 2. Linear / Logistic Regression

**How it works:** Linear Regression fits a straight-line equation — a weighted sum of all
input features (sleep duration, activity level, BMI category, etc.) — to predict a numeric
target. Logistic Regression does the same but converts that weighted sum into a probability
for each class (e.g., probability of "Insomnia" vs. "Sleep Apnea" vs. "None"). Both are
trained by finding the feature weights that minimize prediction error on the training data.

**Purpose in this project:** This is the interpretable middle ground. Because the relationship
is a simple weighted sum, each feature gets one coefficient that says "how much, and in which
direction, does this feature move the prediction?" This is the model type closest to what a
health researcher or reviewer can sanity-check by hand.

**What we can do with it:**
- Report specific, explainable statements, e.g. "each additional hour of sleep is associated
  with an X-point decrease in predicted stress level, holding other factors constant."
- Use it as the "explainable AI" component of the final report/poster, since coefficients
  can be listed in a table with plain-language interpretation.
- Use its performance gap versus Random Forest to check whether relationships in the data are
  mostly linear (small gap) or have important non-linear/interaction effects (large gap).

---

## 3. Random Forest (`RandomForestRegressor` / `RandomForestClassifier`)

**How it works:** A Random Forest builds many decision trees (300–400 in this project), each
trained on a random subset of rows and features. Each tree splits the data repeatedly on
whatever feature best separates high vs. low target values at that point (e.g., "is Sleep
Duration < 6.5?"). The forest's final prediction is the average (regression) or majority vote
(classification) across all trees. Averaging many imperfect, differently-biased trees reduces
overfitting compared to any single tree.

**Purpose in this project:** This is the "best effort" predictive model. It can automatically
capture non-linear effects and interactions (e.g., "low activity only matters for stress when
sleep is also short") that a linear model cannot represent directly.

**What we can do with it:**
- Use its `feature_importances_` output to rank which lifestyle/health factors matter most for
  each target — a natural next analysis step and a strong candidate for a chart in the final
  report.
- Use it as the main predictive model if the project's goal shifts from "explain relationships"
  to "build the most accurate screening/flagging tool."
- Because `class_weight="balanced"` was used for classification, it also correctly handles the
  fact that "Sleep Apnea," "Insomnia," and "None" are not equally common in the data.

---

## Results by Target Variable

All metrics are computed on the same held-out test rows for a given target, so the three
models are directly comparable within each row group. Regression: lower MAE/RMSE and higher
R² are better. Classification: higher accuracy, balanced accuracy, and macro F1 are better.

### Dataset: `Sleep_health_and_lifestyle_dataset.csv` (132 rows after removing exact duplicates, 33 held out for testing)

**Target: Stress Level (0–10 scale)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (always predicts the average) | 1.41 | 1.63 | ~0.00 |
| Linear Regression | 0.30 | 0.89 | 0.71 |
| Random Forest | 0.22 | 0.40 | 0.94 |

*Interpretation:* Both real models dramatically beat the dummy baseline, confirming stress
level is genuinely predictable from sleep and lifestyle features in this dataset. Random
Forest's clear edge over Linear Regression (R² 0.94 vs. 0.71) suggests non-linear or
interaction effects — e.g., stress may not simply scale evenly with sleep duration.

**Target: Quality of Sleep (1–10 scale)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy | 1.04 | 1.19 | ~0.00 |
| Linear Regression | 0.17 | 0.48 | 0.84 |
| Random Forest | 0.11 | 0.29 | 0.94 |

*Interpretation:* Same pattern — sleep quality is well-predicted by the available features,
with Random Forest again ahead, though Linear Regression is already quite strong here.

**Target: Sleep Disorder (None / Insomnia / Sleep Apnea) — 15 test rows**

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---|---|---|
| Dummy (always predicts "None") | 46.7% | 50.0% | 0.32 |
| Logistic Regression | 80.0% | 80.4% | 0.80 |
| Random Forest | 80.0% | 80.4% | 0.80 |

*Interpretation:* Both real models tie and clearly beat guessing the majority class. With
only 15 test rows, treat these numbers as a promising early signal, not a precise estimate —
larger data would be needed to say more confidently which model generalizes better.

### Dataset: `Sleep_Efficiency.csv` (452 rows, 113 or 108 held out for testing depending on target)

**Target: Sleep Efficiency (0–1 scale)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy | 0.120 | 0.139 | ~0.00 |
| Linear Regression | 0.050 | 0.062 | 0.80 |
| Random Forest | 0.038 | 0.051 | 0.86 |

*Interpretation:* Sleep efficiency is well explained by the available factors (sleep stage
percentages, awakenings, caffeine/alcohol, exercise). Random Forest again edges out the linear
model.

**Target: Awakenings (count per night)**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Dummy | 1.20 | 1.37 | ~0.00 |
| Linear Regression | 0.80 | 1.01 | 0.46 |
| Random Forest | 0.73 | 0.88 | 0.58 |

*Interpretation:* This is the weakest-performing target of the five (R² 0.58 at best). Number
of awakenings is harder to predict from the available lifestyle variables — likely because
important drivers (e.g., noise, room temperature, undiagnosed conditions) are not captured in
this dataset.

---

## Important Data Note

The lifestyle dataset dropped from 374 to 132 rows after removing exact duplicate rows (once
the `Person ID` column, which made every row artificially unique, was excluded). This means
many "different people" in the raw file actually share identical values across every other
column. Treat this dataset as a smaller, less varied sample than the original row count
suggests, and mention this explicitly as a limitation in the final report.

## Overall Takeaways for the Final Report

1. **Stress, sleep quality, sleep efficiency, and sleep disorder status** are all meaningfully
   predictable from lifestyle and physiological features — far better than chance.
2. **Awakenings** are the hardest target to predict, suggesting unmeasured factors matter here.
3. **Random Forest generally wins**, but Linear/Logistic Regression is competitive enough to be
   the model you *explain* in the report, while Random Forest is the model you *cite for best
   accuracy* and for feature-importance rankings.
4. All results are **associations in observational data**, not proof of cause and effect — keep
   this limitation in every conclusion you write.

## Suggested Next Step

Add a feature-importance chart from the Random Forest models (highest-impact factors for
stress and sleep quality) — this turns the "black box" model into a clear, visual takeaway for
the community-facing one-pager (e.g., "physical activity and sleep duration were the top two
predictors of lower stress in this data").
