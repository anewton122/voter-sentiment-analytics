"""
analysis.py
===============

This script performs the core analysis for the Voter Sentiment Analytics project.
It reads the survey dataset, computes sampling weights, generates exploratory
charts, computes a correlation matrix, and fits a logistic regression model to
estimate the influence of demographic variables on candidate support.  All
outputs are written to the ``docs/`` directory.

Usage::

    python analysis.py

Ensure that the dataset ``data/survey_data.csv`` exists before running this
script.  If the ``weight`` column is missing, it will be computed based on
gender distribution differences between the sample and a target population.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


def compute_weights(df: pd.DataFrame) -> pd.Series:
    """Compute sampling weights for gender.

    A target population distribution (49% male / 51% female) is used to adjust
    the sample.  Respondents labelled ``Other`` receive a weight of zero in
    this simple example.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a ``gender`` column.

    Returns
    -------
    pandas.Series
        A Series of weights aligned with ``df``.
    """
    # Define a hypothetical population distribution
    pop_gender_dist = {"Male": 0.49, "Female": 0.51, "Other": 0.0}
    # Compute the sample distribution
    sample_gender_counts = df["gender"].value_counts(normalize=True)
    # Map each respondent to a weight based on their gender
    return df["gender"].map(lambda g: pop_gender_dist.get(g, 0) / sample_gender_counts.loc[g])


def generate_bar_chart(series: pd.Series, title: str, filename: str) -> None:
    """Generate a simple bar chart and save it to the docs directory."""
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel("Support Proportion")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main() -> None:
    # Paths
    data_path = os.path.join("data", "survey_data.csv")
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)

    # Read data
    df = pd.read_csv(data_path)

    # Compute weights if not present
    if "weight" not in df.columns:
        df["weight"] = compute_weights(df)

    # Exploratory Data Analysis – support by gender and age
    support_by_gender = df.groupby("gender")["candidate_support"].mean().sort_index()
    support_by_age = df.groupby("age_group")["candidate_support"].mean()
    # Order age groups logically if present
    age_order = ["18-29", "30-44", "45-59", "60+"]
    support_by_age = support_by_age.reindex(age_order)

    # Generate bar charts
    generate_bar_chart(
        support_by_gender,
        title="Candidate Support by Gender",
        filename=os.path.join(docs_dir, "support_by_gender.png"),
    )
    generate_bar_chart(
        support_by_age,
        title="Candidate Support by Age Group",
        filename=os.path.join(docs_dir, "support_by_age.png"),
    )

    # Correlation Matrix
    corr_df = df.copy()
    # Encode categorical variables to numeric codes for correlation
    for col in ["age_group", "gender", "race", "education", "income", "party_affiliation"]:
        le = LabelEncoder()
        corr_df[col] = le.fit_transform(corr_df[col])
    corr_matrix = corr_df.drop(columns=["respondent_id"]).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix of Survey Data (Encoded)")
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, "correlation_matrix.png"))
    plt.close()

    # Logistic Regression using statsmodels
    # One‑hot encode categorical variables (drop_first=True to avoid multicollinearity)
    X = pd.get_dummies(
        df[["age_group", "gender", "race", "education", "income", "party_affiliation"]],
        drop_first=True,
    )
    y = df["candidate_support"]
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    summary_text = result.summary().as_text()
    with open(os.path.join(docs_dir, "logistic_regression_summary.txt"), "w") as f:
        f.write(summary_text)
    print("Analysis complete.  Plots and summary saved to the docs/ directory.")


if __name__ == "__main__":
    main()