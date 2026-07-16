# Sleep, Stress & Lifestyle Insights

A student biomedical research project analyzing two public Kaggle sleep datasets to explore
how sleep habits and lifestyle factors relate to stress, sleep quality, sleep disorders, and
sleep efficiency. Built to share practical, evidence-based takeaways with students and
families, including here in Irvine, CA.

**This is an educational project, not a medical tool.** It does not diagnose conditions or
replace advice from a doctor.

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | Interactive Streamlit website — the thing you deploy online |
| `baseline_models.py` | Command-line script that trains all models and saves results/charts to `outputs/` |
| `Model_Report.md` | Full written report explaining each model and the findings |
| `Sleep_health_and_lifestyle_dataset.csv` | Kaggle dataset 1 (sleep, stress, activity, BMI, blood pressure, sleep disorder) |
| `Sleep_Efficiency.csv` | Kaggle dataset 2 (sleep efficiency, sleep stages, caffeine/alcohol, exercise) |
| `outputs/` | Saved cleaned data, results table, and chart from `baseline_models.py` |
| `requirements.txt` | Python packages needed to run the app |

`dataset_2191_sleep.csv` (animal sleep data) is kept for reference but is intentionally not
used in any analysis.

## Run it on your own computer first

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

This opens the site at `http://localhost:8501` in your browser.

## Deploy it online for free (Streamlit Community Cloud)

You said you already have a GitHub account, so here's the fastest path to a permanent public
link:

### 1. Push this folder to GitHub

Open a terminal in this project folder and run:

```powershell
git add -A
git commit -m "Deploy sleep, stress, and lifestyle insights app"
```

(The repo has already been initialized and the first commit made for you.)

Then create a new, empty repository on GitHub (no README/license, so it stays empty):
1. Go to [github.com/new](https://github.com/new)
2. Name it something like `sleep-stress-lifestyle-insights`
3. Keep it **Public** (required for the free Streamlit Cloud tier) and click **Create repository**
4. Copy the commands GitHub shows you under "…or push an existing repository", which will look like:

```powershell
git remote add origin https://github.com/YOUR-USERNAME/sleep-stress-lifestyle-insights.git
git branch -M main
git push -u origin main
```

Run those in this project folder. GitHub will prompt you to sign in the first time.

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
2. Click **Create app** → **Deploy a public app from GitHub**
3. Pick your new repository, branch `main`, and main file path `app.py`
4. Click **Deploy**

After a minute or two, you'll get a permanent public URL like
`https://your-app-name.streamlit.app` that you can share with teachers, classmates, or your
community — it stays online even when your own computer is off.

### 3. Updating the site later

Any time you change the code or data, run:

```powershell
git add -A
git commit -m "Describe your change"
git push
```

Streamlit Community Cloud automatically redeploys the site within a minute of the push.

## Data source and limitations

- Both datasets are public, de-identified, and downloaded from Kaggle. Neither was collected
  in Irvine, CA — treat all findings as general patterns to inform recommendations, not as
  data about the local community specifically.
- All results describe **associations**, not proof that one factor **causes** another.
- See `Model_Report.md` for the full methodology, results, and a documented data-cleaning fix
  (pandas was initially misreading the literal text "None" in the Sleep Disorder column as a
  missing value; this has been corrected).
