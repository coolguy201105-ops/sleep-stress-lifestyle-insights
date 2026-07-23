# Research Summary: Sleep, Stress & Lifestyle Insights

**A Data-Driven Biomedical Research Project for the Irvine Community**

## 1. Background & Motivation

Poor sleep and elevated stress are widespread, everyday health concerns for students and
families — linked to reduced academic performance, weakened immune function, and long-term
cardiovascular and mental-health risk. Local, Irvine-specific health survey data is not
publicly available, but national and international sleep research consistently identifies the
same risk factors (short sleep, high stress, poor sleep quality) that likely also affect our
community. This project analyzes public, de-identified biomedical datasets to quantify which
lifestyle and physiological factors are most strongly associated with poor sleep and elevated
stress, then turns those findings into a practical, shareable web tool and a set of plain-
language recommendations.

## 2. Research Question & Hypothesis

**Question:** How are sleep duration, sleep quality, and lifestyle factors (physical activity,
caffeine, alcohol, exercise, BMI) associated with stress level, sleep disorders, and sleep
efficiency?

**Hypothesis:** Individuals with shorter or lower-quality sleep will show higher stress levels,
higher rates of diagnosed sleep disorders, and lower sleep efficiency than individuals with
longer, higher-quality sleep.

## 3. Data Sources

Five public Kaggle datasets were collected and reviewed; two were used for modeling.

| Dataset | Size | Used? | What it provides |
|---|---|---|---|
| Sleep Health & Lifestyle | 374 people × 13 fields | Yes — main dataset | Sleep duration/quality, physical activity (minutes/day of exercise), stress level, BMI, blood pressure, heart rate, daily steps, diagnosed sleep disorder |
| Sleep Efficiency | 452 people × 15 fields | Yes — supporting dataset | Bedtime/wake time, sleep efficiency %, sleep-stage percentages, awakenings, caffeine/alcohol intake, smoking status, exercise frequency |
| Animal sleep dataset | 62 species × 8 fields | No — excluded | Non-human (body/brain weight, lifespan, predation risk) — not relevant to a human health study |
| Wearable tech sleep quality | 1,000 rows × 9 fields | No — reviewed, not used | Simulated sensor data (heart-rate variability, movement, body temperature) with no demographic fields; appears synthetic |
| Health Sleep Statistics | 100 people × 12 fields | No — reviewed, not used | Overlaps the main dataset's fields but has a much smaller sample and no stress-level column |

The two datasets used were always analyzed **separately** (never merged row-by-row), since they
were collected independently with different columns and sample sizes.

## 4. Methods

1. **Data cleaning** — removed exact duplicate records, fixed a data-loading bug where the text
   `"None"` in the Sleep Disorder column was silently read as a missing value, converted
   bedtime/wake-time text into numeric clock minutes, and standardized column names.
2. **Exploratory analysis** — visualized relationships between sleep duration, stress, and sleep
   quality across the sample.
3. **Predictive modeling** — for five target variables (stress level, sleep quality, sleep
   disorder, sleep efficiency, awakenings), trained and compared three models of increasing
   complexity on an identical 75/25 train-test split:
   - **Dummy baseline** — predicts the average (regression) or most common class
     (classification); the "floor" every real model must beat.
   - **Linear / Logistic Regression** — an interpretable, fully explainable model.
   - **Random Forest** — a flexible model capturing non-linear effects and interactions.
4. **Interpretation** — ranked Random Forest feature importances to identify the strongest
   predictors for each outcome.
5. **Deployment** — packaged everything into a public, interactive Streamlit web app so anyone
   can explore the data, compare model accuracy, and test their own lifestyle inputs.

## 5. Key Results

| Target | Best Model | Best Score | vs. Dummy Baseline |
|---|---|---|---|
| Stress Level (0–10) | Random Forest | R² = 0.94 | Error cut by ~84% |
| Quality of Sleep (1–10) | Random Forest | R² = 0.94 | Error cut by ~89% |
| Sleep Disorder (3-class) | Random Forest | 66.7% accuracy, 62.2% balanced accuracy | +12 pts accuracy, +29 pts balanced accuracy |
| Sleep Efficiency (0–1) | Random Forest | R² = 0.86 | Error cut by ~68% |
| Awakenings (count/night) | Random Forest | R² = 0.58 | Error cut by ~39% |

**Top predictors, by Random Forest importance:**

- **Stress level:** self-rated sleep quality, resting heart rate, and sleep duration are the
  strongest signals; age and daily steps contribute modestly.
- **Sleep quality:** sleep duration is by far the dominant predictor, followed by stress level
  and heart rate.
- **Sleep disorder status:** the signal is more spread out — sleep duration, age, heart rate,
  physical activity level, and daily steps are all similarly important, which matches the more
  modest accuracy for this target.
- **Sleep efficiency & awakenings:** driven mostly by sleep-architecture measurements themselves
  (light/deep sleep percentages, prior awakenings) rather than lifestyle choices — caffeine,
  alcohol, and smoking mattered less than expected, suggesting sleep *structure* dominates over
  the specific lifestyle inputs captured in this dataset.

*Note:* Stress level and sleep quality are each other's strongest predictor, which is expected
(they are closely linked subjective ratings) but means part of that "predictive power" reflects
shared subjective rating bias, not purely independent lifestyle causes.

## 6. Interpretation & Community Takeaways

- **Sleep duration and quality are the single biggest levers** for lower stress in this data —
  more consistent, higher-quality sleep is associated with meaningfully lower stress ratings.
- **Sleep disorder risk is harder to flag from lifestyle data alone** (mid-60% accuracy on three
  classes) — useful as an early, non-diagnostic signal, not a substitute for clinical screening.
- **Sleep efficiency problems** (frequent awakenings, low deep/light sleep balance) may need
  different interventions than simple lifestyle changes, since caffeine/alcohol/exercise showed
  smaller effects than expected.
- All three "real" models beat the naive baseline by wide margins for four of five targets,
  confirming these relationships are genuine patterns in the data, not noise.

## 7. Limitations & Ethics

- Datasets are **public and not collected in Irvine** — findings represent general patterns
  from these samples, not verified local statistics.
- All datasets are **observational**: results describe *associations*, not proof that one
  factor *causes* another.
- The main lifestyle dataset shrank from 374 to 132 unique rows after removing exact
  duplicates, meaning the effective sample is smaller and less varied than it first appears.
- This project and its web predictor are **educational tools only** — not a diagnostic or
  medical device, and they do not replace advice from a healthcare professional.

## 8. Deliverables

- A public, interactive **Streamlit web app** — explore the data, compare model accuracy across
  all three model types, and test personal lifestyle inputs against the trained models.
- A **full written model report** (`Model_Report.md`) documenting methodology, per-target
  results, and every dataset considered.
- A **one-page project proposal** and this **research summary** for presentation and sharing.
- This **slide deck** for presenting findings to the class/community.

## 9. Conclusion & Next Steps

This project shows that everyday, easily self-reported lifestyle and physiological factors —
especially sleep duration and quality — carry a strong, measurable relationship with stress and
sleep health outcomes. The next steps are to (1) validate these patterns against a larger or
local sample if one becomes available, (2) extend the wearable-sensor dataset into the modeling
pipeline once it can be paired with real demographic data, and (3) turn the strongest, most
actionable findings (sleep duration, consistency, and stress) into a one-page community handout
for Irvine students and families.
