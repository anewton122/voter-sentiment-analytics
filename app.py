"""
app.py
======

Streamlit application for the Voter Sentiment Analytics project.
The app allows users to filter the survey data by demographic variables,
examine support rates for a hypothetical candidate, visualize correlations,
and inspect logistic regression coefficients.  To run the app locally::

    streamlit run src/app.py

The dataset ``data/survey_data.csv`` must be present in the repository root.
"""

import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data(path: str = "data/survey_data.csv") -> pd.DataFrame:
    """Load the survey dataset from a CSV file."""
    return pd.read_csv(path)


@st.cache_data
def fit_logit_model(df: pd.DataFrame) -> sm.Logit:
    """
    Fit a logistic regression model to the survey data.

    Categorical variables are one‑hot encoded (dropping the first level to avoid
    multicollinearity).  A constant intercept term is added.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered subset of the survey data.

    Returns
    -------
    result : statsmodels.api.Logit
        Fitted logistic regression result object.
    """
    X = pd.get_dummies(
        df[["age_group", "gender", "race", "education", "income", "party_affiliation"]],
        drop_first=True,
    )
    y = df["candidate_support"]
    X = sm.add_constant(X)
    model = sm.Logit(y, X)
    result = model.fit(disp=0)
    return result


def main() -> None:
    st.set_page_config(page_title="Voter Sentiment Analytics", layout="wide")
    st.title("Voter Sentiment Analytics Dashboard")

    # Load data
    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    sel_age = st.sidebar.multiselect(
        "Age Group", df["age_group"].unique().tolist(), default=df["age_group"].unique().tolist()
    )
    sel_gender = st.sidebar.multiselect(
        "Gender", df["gender"].unique().tolist(), default=df["gender"].unique().tolist()
    )
    sel_race = st.sidebar.multiselect(
        "Race", df["race"].unique().tolist(), default=df["race"].unique().tolist()
    )
    sel_edu = st.sidebar.multiselect(
        "Education", df["education"].unique().tolist(), default=df["education"].unique().tolist()
    )
    sel_income = st.sidebar.multiselect(
        "Income", df["income"].unique().tolist(), default=df["income"].unique().tolist()
    )
    sel_party = st.sidebar.multiselect(
        "Party Affiliation", df["party_affiliation"].unique().tolist(), default=df["party_affiliation"].unique().tolist()
    )

    # Apply filters
    filtered = df[
        (df["age_group"].isin(sel_age))
        & (df["gender"].isin(sel_gender))
        & (df["race"].isin(sel_race))
        & (df["education"].isin(sel_edu))
        & (df["income"].isin(sel_income))
        & (df["party_affiliation"].isin(sel_party))
    ]

    # Display summary statistic
    support_rate = filtered["candidate_support"].mean() if len(filtered) > 0 else 0
    st.metric("Support Rate", f"{support_rate:.1%}")

    # Display bar charts
    # Support by gender
    st.subheader("Support by Gender")
    gender_support = (
        filtered.groupby("gender")["candidate_support"].mean().sort_index() if len(filtered) > 0 else pd.Series()
    )
    if not gender_support.empty:
        st.bar_chart(gender_support)
    else:
        st.write("No data for selected filters.")

    # Support by age group
    st.subheader("Support by Age Group")
    age_order = ["18-29", "30-44", "45-59", "60+"]
    age_support = filtered.groupby("age_group")["candidate_support"].mean().reindex(age_order)
    if not age_support.empty:
        st.bar_chart(age_support)
    else:
        st.write("No data for selected filters.")

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    if len(filtered) > 1:
        corr_df = filtered.copy()
        for col in ["age_group", "gender", "race", "education", "income", "party_affiliation"]:
            corr_df[col] = pd.factorize(corr_df[col])[0]
        corr_matrix = corr_df.drop(columns=["respondent_id"]).corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough data points to compute a correlation matrix.")

    # Logistic regression coefficients
    st.subheader("Logistic Regression Coefficients")
    if len(filtered) > 1:
        result = fit_logit_model(filtered)
        coef_df = result.params.to_frame(name="Coefficient").reset_index().rename(columns={"index": "Variable"})
        st.dataframe(coef_df)
        st.write(
            "Coefficients are estimated from a logistic regression model.  Positive values increase the log‑odds of supporting the candidate, negative values decrease them."
        )
    else:
        st.write("Not enough data points to fit a model.")


if __name__ == "__main__":
    main()