# Voter Sentiment Analytics Platform

This repository demonstrates an end‑to‑end workflow for analyzing voter sentiment data, inspired by professional work carried out at **GPSP** and **Advanced Insights**.  Our small team at GPSP conducted partisan and non‑partisan polling for Senate campaigns, lobbyists and city governments.  My role was to turn raw survey responses into actionable insights: cleaning data, applying demographic weights, building predictive models, and packaging the results for non‑technical stakeholders.  This project recreates that workflow in a self‑contained, open‑source example.

## Dataset

The repository includes a synthetic survey dataset (`data/survey_data.csv`) containing 1,000 observations.  Each row represents a survey respondent with the following fields:

| Field                | Description                                                                                             |
|----------------------|---------------------------------------------------------------------------------------------------------|
| `respondent_id`      | Unique identifier for each respondent                                                                     |
| `age_group`          | Age bracket (`18–29`, `30–44`, `45–59`, `60+`)                                                            |
| `gender`             | Respondent gender (`Male`, `Female`, `Other`)                                                             |
| `race`               | Race/ethnicity (`White`, `Black`, `Hispanic`, `Asian`, `Other`)                                           |
| `education`          | Highest educational attainment (`High School or Less`, `Some College/Associates`, `Bachelor`, `Graduate`) |
| `income`             | Income category (`Low`, `Medium`, `High`)                                                                |
| `party_affiliation`  | Political affiliation (`Democrat`, `Republican`, `Independent`, `Other`)                                 |
| `candidate_support`  | Binary indicator (1 = supports the candidate, 0 = does not)                                              |
| `weight`             | Sampling weight to adjust the sample’s gender distribution to match a target population                   |

While the dataset is synthetic, the structure mirrors typical voter opinion polls.  You can substitute your own survey data in place of this file—ensure the column names match to reuse the analysis and dashboard code.

## Analysis Workflow

The core analysis is contained in the script **`analysis.py`**, which performs the following steps:

1. **Data Cleaning & Weighting** – The script reads the CSV, computes sampling weights based on gender distribution differences between the sample and an assumed population (49% male / 51% female), and stores the weighted dataset for downstream use.
2. **Exploratory Data Analysis (EDA)** – Summary statistics and charts are generated to understand how candidate support varies by gender and age group.  The resulting charts are saved in the `docs/` directory (e.g. `support_by_gender.png`, `support_by_age.png`).
3. **Correlation Analysis** – Categorical variables are encoded to numeric codes and a correlation matrix is computed and visualized.  The heatmap is saved as `docs/correlation_matrix.png`.
4. **Logistic Regression Modeling** – Using [StatsModels](https://www.statsmodels.org/), a logistic regression is fit on one‑hot encoded features to estimate the effect of demographic variables on candidate support.  A text summary of the model is saved to `docs/logistic_regression_summary.txt`.

## Interactive Dashboard

To make the analysis accessible to non‑technical stakeholders (e.g. campaign staff or city officials), the repository includes a Streamlit app in **`src/app.py`**.  When run, the app lets users:

* Filter respondents by demographic attributes such as gender, age group, race, education, income and party affiliation.
* View updated bar charts showing the proportion of respondents who support the candidate within the selected group.
* Inspect the logistic regression coefficients and their significance.
* Review the correlation heatmap for the filtered dataset.

Although the Streamlit app cannot be run in this environment, the code is production‑ready and can be executed locally or deployed to [Streamlit Cloud](https://streamlit.io/cloud).  See the **Getting Started** section below for instructions.

## Getting Started

1. **Clone the repository** and install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the analysis script** to reproduce the charts and model summary:

   ```bash
   python analysis.py
   ```

   The script will read `data/survey_data.csv`, compute weights, perform EDA, generate plots in `docs/`, fit a logistic regression model, and write its summary to `docs/logistic_regression_summary.txt`.

3. **Launch the dashboard** to interactively explore the data:

   ```bash
   streamlit run src/app.py
   ```

   A local Streamlit server will start.  Use the sidebar filters to select demographic groups and observe how candidate support changes.

## File Structure

```
voter-sentiment-analytics/
├── data/
│   └── survey_data.csv        # Synthetic survey responses with demographic attributes, support indicator and weights
├── docs/
│   ├── correlation_matrix.png # Correlation heatmap produced by analysis.py
│   ├── support_by_age.png     # Bar chart of support by age group
│   ├── support_by_gender.png  # Bar chart of support by gender
│   └── logistic_regression_summary.txt # StatsModels summary of the logistic regression model
├── src/
│   └── app.py                 # Streamlit application for interactive exploration
├── analysis.py                # Analysis script: cleaning, weighting, EDA, correlation, modeling
├── requirements.txt           # Python dependencies
└── README.md                  # Project description and usage
```

## Professional Context

At **GPSP**, our team of four conducted partisan polling for political campaigns, including Senate races like the Reid Rasner campaign in Wyoming.  We were responsible for sending surveys into the field, weighting responses to match population demographics, building correlation matrices and tables, and delivering insights to campaign strategists.  After transitioning to **Advanced Insights**, we applied the same methodology to non‑partisan projects for lobbyists and municipal governments, such as gauging public sentiment on eminent domain in Iowa or assessing attitudes toward public transportation in San Antonio.

This project mirrors that real‑world workflow.  It demonstrates how to:

* Build a **reproducible data pipeline** that ingests raw survey data, applies demographic weights and performs cleaning.
* Conduct **exploratory analysis** and compute **correlations** to identify which demographic factors move the needle.
* Fit a **logistic regression model** to quantify the influence of each variable on candidate support.
* Package results into an **interactive dashboard** so decision‑makers can slice the data by demographic groups and understand what drives sentiment.

By publishing this project, I showcase my ability to combine **data science**, **software engineering**, and **DevOps** practices into a polished portfolio piece suitable for data analyst, data scientist, software engineer or DevOps roles.