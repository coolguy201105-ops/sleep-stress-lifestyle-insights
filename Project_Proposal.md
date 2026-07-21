# Project Proposal: Sleep, Stress & Lifestyle Insights

**A Data-Driven Biomedical Research Project for the Irvine Community**

## Problem & Community Relevance
Poor sleep and high stress are widespread concerns among students and families, contributing
to reduced academic performance, weakened immune function, and long-term health risks. While
Irvine-specific health data is not publicly available, national and international sleep
research consistently shows the same risk factors likely affect local students and families.
This project analyzes public biomedical datasets to identify which lifestyle and physiological
factors most strongly relate to poor sleep and elevated stress, then translates those findings
into a practical, shareable tool and recommendations for the Irvine community.

## Research Question
How are sleep duration, sleep quality, and lifestyle factors (physical activity, caffeine,
alcohol, exercise, BMI) associated with stress level, sleep disorders, and sleep efficiency?

**Hypothesis:** Individuals with shorter or lower-quality sleep will show higher stress levels,
higher rates of sleep disorders, and lower sleep efficiency than those with longer, higher-
quality sleep.

## Data Sources
Two public, de-identified Kaggle datasets (an unrelated animal-sleep dataset was excluded):
1. **Sleep Health & Lifestyle Dataset** (374 records) — sleep duration/quality, stress level,
   physical activity, BMI, blood pressure, heart rate, daily steps, sleep disorder diagnosis.
2. **Sleep Efficiency Dataset** (452 records) — sleep duration/efficiency, sleep-stage
   percentages, awakenings, caffeine/alcohol use, smoking status, exercise frequency.

## Methods
1. **Data cleaning** — remove duplicates, fix data-parsing errors, handle missing values.
2. **Exploratory analysis** — visualize relationships (e.g., stress by sleep-duration group).
3. **Predictive modeling** — for each target (stress, sleep quality, sleep disorder, sleep
   efficiency, awakenings), train and compare a naive baseline, an interpretable
   linear/logistic regression, and a random forest, evaluated on held-out test data.
4. **Interpretation** — use feature-importance rankings to identify the strongest predictors.

## Deliverables
- A public, interactive website (Streamlit, already deployed) for exploring the data, comparing
  model accuracy, and testing personal lifestyle inputs against the models.
- A full written report documenting methodology, results, and limitations.
- A one-page recommendation sheet for Irvine students and families on sleep and stress.

## Ethics & Limitations
- Data is public and not collected in Irvine — findings represent general patterns, not local
  statistics. All results describe **associations**, not proof of cause and effect.
- The project and predictor tool are educational only, not a diagnostic or medical tool.

## Timeline
| Week | Task |
|---|---|
| 1 | Finalize datasets, clean and prepare data |
| 2 | Exploratory analysis and visualization |
| 3 | Build and evaluate baseline models |
| 4 | Build interactive web app; deploy online |
| 5 | Write final report and community recommendation sheet; present findings |
