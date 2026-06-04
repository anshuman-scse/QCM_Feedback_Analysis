import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Student Feedback Analytics",
    layout="wide"
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IGNORE_PATTERNS = [
    "timestamp",
    "email",
    "name",
    "student",
    "roll",
    "registration",
    "enrollment",
    "mobile",
    "phone",
    "contact"
]

EXCLUDE_COMMENTS = {
    "no",
    "nothing",
    "nil",
    "na",
    "n/a",
    "-",
    ".",
    ""
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def should_ignore(col):

    col = str(col).lower()

    return any(
        pattern in col
        for pattern in IGNORE_PATTERNS
    )

def extract_score(value):

    if pd.isna(value):
        return np.nan

    match = re.match(r"(\d+)", str(value))

    if match:
        return int(match.group(1))

    return np.nan

def detect_rating_columns(df):

    rating_cols = []

    for col in df.columns:

        if should_ignore(col):
            continue

        scores = df[col].apply(extract_score)

        valid_ratio = scores.notna().mean()

        if valid_ratio > 0.70:
            rating_cols.append(col)

    return rating_cols

def detect_text_columns(df):

    text_cols = []

    for col in df.columns:

        if should_ignore(col):
            continue

        values = (
            df[col]
            .dropna()
            .astype(str)
        )

        if len(values) == 0:
            continue

        avg_len = values.str.len().mean()

        if avg_len > 10:
            text_cols.append(col)

    return text_cols

def sentiment_label(score):

    if score > 0.1:
        return "Positive"

    elif score < -0.1:
        return "Negative"

    return "Neutral"

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Student Feedback Analytics Dashboard")

uploaded_files = st.file_uploader(
    "Upload QCM / Feedback Excel Files",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:

    all_ratings = []
    all_comments = []

    survey_summary = []

    # --------------------------------------------
    # PROCESS FILES
    # --------------------------------------------

    for file in uploaded_files:

        df = pd.read_excel(file)

        survey_name = file.name.replace(".xlsx", "")

        rating_cols = detect_rating_columns(df)

        text_cols = detect_text_columns(df)

        rating_df = pd.DataFrame()

        for col in rating_cols:
            rating_df[col] = df[col].apply(extract_score)

        rating_df["Survey"] = survey_name

        all_ratings.append(rating_df)

        survey_avg = (
            rating_df
            .drop(columns=["Survey"])
            .mean()
            .mean()
        )

        survey_summary.append({
            "Survey": survey_name,
            "AverageScore": round(survey_avg,2),
            "Responses": len(df)
        })

        for col in text_cols:

            comments = (
                df[col]
                .dropna()
                .astype(str)
            )

            comments = comments[
                ~comments.str.lower().isin(
                    EXCLUDE_COMMENTS
                )
            ]

            for c in comments:

                sentiment = TextBlob(c).sentiment.polarity

                all_comments.append({
                    "Survey": survey_name,
                    "Question": col,
                    "Comment": c,
                    "SentimentScore": sentiment,
                    "Sentiment": sentiment_label(sentiment)
                })

    rating_data = pd.concat(
        all_ratings,
        ignore_index=True
    )

    comments_df = pd.DataFrame(all_comments)

    summary_df = pd.DataFrame(survey_summary)

    # --------------------------------------------
    # KPIs
    # --------------------------------------------

    st.header("Overview")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Responses",
        int(summary_df["Responses"].sum())
    )

    c2.metric(
        "Surveys",
        len(summary_df)
    )

    c3.metric(
        "Overall Avg Score",
        round(
            summary_df["AverageScore"].mean(),
            2
        )
    )

    if len(comments_df):

        positive_pct = round(
            (
                comments_df["Sentiment"]
                .eq("Positive")
                .mean()
            ) * 100,
            1
        )

        c4.metric(
            "Positive Sentiment %",
            positive_pct
        )

    st.divider()

    # --------------------------------------------
    # SURVEY TREND
    # --------------------------------------------

    st.subheader("Survey Trend")

    fig = px.line(
        summary_df,
        x="Survey",
        y="AverageScore",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------
    # QUESTION-WISE ANALYSIS
    # --------------------------------------------

    st.subheader("Question-wise Ratings")

    question_summary = (
        rating_data
        .drop(columns=["Survey"])
        .mean()
        .sort_values(ascending=False)
    )

    question_df = pd.DataFrame({
        "Question": question_summary.index,
        "Average Score": question_summary.values
    })

    fig = px.bar(
        question_df,
        x="Average Score",
        y="Question",
        orientation="h",
        height=900
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------
    # SENTIMENT
    # --------------------------------------------

    if len(comments_df):

        st.subheader("Sentiment Analysis")

        sentiment_counts = (
            comments_df["Sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "Sentiment",
            "Count"
        ]

        fig = px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------
    # WORD CLOUD
    # --------------------------------------------

    if len(comments_df):

        st.subheader("Word Cloud")

        text = " ".join(
            comments_df["Comment"]
        )

        wc = WordCloud(
            width=1200,
            height=500,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)

    # --------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------

    st.subheader("Executive Summary")

    overall_score = round(
        summary_df["AverageScore"].mean(),
        2
    )

    positive = len(
        comments_df[
            comments_df["Sentiment"]=="Positive"
        ]
    )

    negative = len(
        comments_df[
            comments_df["Sentiment"]=="Negative"
        ]
    )

    st.info(
        f"""
Overall Satisfaction Score: {overall_score}/5

Positive Comments: {positive}

Negative Comments: {negative}

The student feedback indicates generally positive perceptions of the teaching-learning process.
Survey trends, sentiment patterns and open-ended responses should be reviewed together while identifying strengths and improvement areas.
"""
    )