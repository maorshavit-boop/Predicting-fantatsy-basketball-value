# 🏀 NBA Fantasy Value Predictor (9-Cat)

## Overview
Success in 9-category Head-to-Head (H2H) fantasy basketball requires more than just picking good scorers. It requires balancing nine distinct statistical categories (**PTS, REB, AST, STL, BLK, 3PM, FG%, FT%, TO**).

This project builds a Machine Learning pipeline to **predict a player's future fantasy value (Z-score)** based on their past performance. The goal is to identify undervalued players and support data-driven drafting decisions for the upcoming season.

## Objectives
* **Predict 9-Cat Value:** Forecast the total Z-score for NBA players for the upcoming season.
* **Beat the Baseline:** Outperform the naive prediction method (assuming next season's stats = last season's stats).
* **Feature Engineering:** Analyze how rolling averages and age curves impact performance.

## Tech Stack
* **Language:** Python 3.9
* **Libraries:** Pandas, NumPy, Scikit-Learn, Statsmodels, XGBoost, Matplotlib/Seaborn
* **Data Source:** NBA API & Basketball Reference (2020-2025 seasons)

## Methodology

### 1. Data Collection & Cleaning
* Extracted per-game stats for ~2000 players across 5 NBA seasons.
* Normalized data using **Z-scores** relative to the league average for each season.
* Handled missing data and filtered for players with significant playing time (>50 career games between 2020-2025).

### 2. Feature Engineering
* **Rolling Averages:** Created 2-year rolling averages to capture recent form.
* **Deltas:** Calculated year-over-year changes to detect improvement/decline trends.
* **Demographics:** Included `Age` and `Age²` (quadratic term) to model the physical peak of athletes.

### 3. Modeling
I framed this as a regression problem, testing multiple models to predict the `Next Season Total Z-Score`.

| Model | MAE (Mean Abs Error) | R² Score | Notes |
| :---- | :---- | :---- | :---- |
| **OLS (Statsmodels)** | **1.750** | **0.714** | **Best Performer.** Used backward elimination for feature selection. |
| Random Forest | 1.816 | 0.692 | Good at capturing non-linearities but slightly overfitted. |
| XGBoost | 1.801 | 0.686 | Robust, but slightly behind OLS in this specific dataset. |
| *Baseline* | *1.857* | *0.669* | *Naive prediction (Next Year = Last Year).* |

## Key Findings
1.  **OLS Supremacy:** Surprisingly, the simple Ordinary Least Squares (OLS) model performed best. This suggests that the relationship between past and future stats is largely linear.
2.  **The "Scoring" Bias:** Feature importance analysis showed that **Points (PTS)** and **Points Per 36** were the strongest predictors of future fantasy success.
3.  **Predictability:** Calculating the "stability" of stats showed that **Assists** and **Rebounds** are highly predictable year-over-year, while **FG%** and **FT%** are volatile and harder to forecast.

## How to run
1. Clone the repo by - git clone [https://github.com/maorshavit-boop/Predicting-fantatsy-basketball-value.git](https://github.com/maorshavit-boop/Predicting-fantatsy-basketball-value.git)
2. Install dependencies by - pip install -r requirements.txt
3. Open the notebook in Jupyter Lab/Notebook

## Project Structure
```text
├── data/               # Raw and processed CSV data
├── plots/              # Visualizations for the README
├── code/               # Jupyter Notebooks with full analysis
├── requirements.txt    # Python dependencies
├── .gitignore          
└── README.md           # Project overview
