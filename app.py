
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO
import re

st.set_page_config(page_title="Student Feedback Analytics", layout="wide")

IGNORE_PATTERNS = [
    "timestamp","email","name","student","roll",
    "registration","enrollment","mobile","phone","contact"
]

from wordcloud import STOPWORDS as WC_STOPWORDS

STOPWORDS = set(WC_STOPWORDS)

STOPWORDS.update({

    "strongly","agree","disagree","neutral",

    "course","courses",
    "faculty","student","students",
    "subject","learning",

    "would","could","should",
    "also","more","one",

    "the","and","for","with","from",
    "that","this","these","those",

    "have","has","had","been",
    "were","was","are","is",

    "into","onto","over","under",

    "please","thank","thanks",

    "good","excellent",

    "objectives","outcomes",

    "qcm","feedback"
})

EXCLUDE_COMMENTS = {"", ".", "-", "na", "n/a", "nil", "nothing", "none", "no"}

def survey_order(name):
    n = name.lower()
    if "qcm-1" in n or "qcm1" in n: return 1
    if "qcm-2" in n or "qcm2" in n: return 2
    if "qcm-3" in n or "qcm3" in n: return 3
    if "feedback" in n: return 4
    return 99

def clean_name(name):
    n=name.lower()
    if "qcm-1" in n or "qcm1" in n: return "QCM-1"
    if "qcm-2" in n or "qcm2" in n: return "QCM-2"
    if "qcm-3" in n or "qcm3" in n: return "QCM-3"
    if "feedback" in n: return "Final Feedback"
    return name

def should_ignore(col):
    c=str(col).lower()
    return any(x in c for x in IGNORE_PATTERNS)

def extract_score(v):
    m=re.match(r"(\d+)", str(v))
    return int(m.group(1)) if m else np.nan

def detect_rating_columns(df):
    cols=[]
    for c in df.columns:
        if should_ignore(c): continue
        ratio=df[c].apply(extract_score).notna().mean()
        if ratio>0.7:
            cols.append(c)
    return cols

def detect_text_columns(df):

    text_cols = []

    OPEN_ENDED_KEYWORDS = [
        "what",
        "suggest",
        "improve",
        "challenge",
        "support",
        "effective",
        "helped",
        "feedback",
        "concern",
        "highlight",
        "aspect"
    ]

    for col in df.columns:

        if should_ignore(col):
            continue

        col_lower = str(col).lower()

        if any(
            keyword in col_lower
            for keyword in OPEN_ENDED_KEYWORDS
        ):
            text_cols.append(col)

    return text_cols

st.title("Student Feedback Analytics Dashboard")

uploads=st.file_uploader("Upload QCM / Feedback Excel Files",
                         type=["xlsx"],accept_multiple_files=True)

if uploads:
    survey_rows=[]
    all_comments=[]
    all_questions=[]

    for f in uploads:
        df=pd.read_excel(f)
        survey=clean_name(f.name)

        ratings=detect_rating_columns(df)
        texts=detect_text_columns(df)

        rdf=pd.DataFrame()
        for c in ratings:
            rdf[c]=df[c].apply(extract_score)

        avg=float(rdf.mean().mean()) if not rdf.empty else np.nan

        survey_rows.append({
            "Survey":survey,
            "AverageScore":round(avg,2),
            "Responses":len(df),
            "Order":survey_order(f.name)
        })

        for c in ratings:
            all_questions.append({
                "Survey":survey,
                "Question":c,
                "Score":rdf[c].mean()
            })

        for c in texts:
            comments=df[c].dropna().astype(str)
            comments=comments[~comments.str.lower().isin(EXCLUDE_COMMENTS)]

            for txt in comments:
                pol=TextBlob(txt).sentiment.polarity
                all_comments.append({
                    "Survey":survey,
                    "Question":c,
                    "Comment":txt,
                    "Sentiment":("Positive" if pol>0.1 else "Negative" if pol<-0.1 else "Neutral"),
                    "Score":pol
                })

    summary=pd.DataFrame(survey_rows).sort_values("Order")
    questions=pd.DataFrame(all_questions)
    comments=pd.DataFrame(all_comments)

    c1,c2,c3=st.columns(3)
    c1.metric("Total Responses", int(summary["Responses"].sum()))
    c2.metric("Overall Average", round(summary["AverageScore"].mean(),2))
    if not comments.empty:
        c3.metric("Positive %", round((comments["Sentiment"]=="Positive").mean()*100,1))

    st.subheader("QCM / Feedback Trend")
    fig=px.line(summary,x="Survey",y="AverageScore",markers=True)
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Question-wise Ratings")
    if not questions.empty:
        q=questions.groupby("Question")["Score"].mean().reset_index()
        q=q.sort_values("Score")
        fig=px.bar(q,x="Score",y="Question",orientation="h",height=800)
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Sentiment Analysis")
    if not comments.empty:
        s=comments["Sentiment"].value_counts().reset_index()
        s.columns=["Sentiment","Count"]
        fig=px.pie(s,names="Sentiment",values="Count")
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Word Clouds by Feedback Question")

    if not comments.empty:
    
        for survey in summary["Survey"]:
    
            st.markdown(f"## {survey}")
    
            survey_comments = comments[
                comments["Survey"] == survey
            ]
    
            if survey_comments.empty:
                continue
    
            for question in survey_comments["Question"].unique():
    
                st.markdown(f"**{question}**")
    
                text = " ".join(
                    survey_comments[
                        survey_comments["Question"] == question
                    ]["Comment"]
                )
    
                text = text.strip()
    
                if len(text) < 5:
    
                    st.info(
                        "Insufficient textual feedback available."
                    )
    
                    continue
    
                try:
    
                    wc = WordCloud(
                        width=1000,
                        height=400,
                        background_color="white",
                        stopwords=STOPWORDS
                    ).generate(text)
    
                    fig, ax = plt.subplots(
                        figsize=(10, 4)
                    )
    
                    ax.imshow(wc)
    
                    ax.axis("off")
    
                    st.pyplot(fig)
    
                except ValueError:
    
                    st.info(
                        "No meaningful words available for this question."
                    )
    st.subheader("Executive Summary")

    if not comments.empty:
        words=[]
        for txt in comments["Comment"]:
            for w in re.findall(r"[A-Za-z]+", txt.lower()):
                if len(w)>3 and w not in STOPWORDS:
                    words.append(w)

        common=Counter(words).most_common(15)

        strengths=[w for w,_ in common[:5]]
        improvements=[w for w,_ in common[5:10]]
        terms=[w for w,_ in common[:10]]

        a,b,c=st.columns(3)

        with a:
            st.markdown("### Top Strengths")
            for x in strengths:
                st.write("•",x)

        with b:
            st.markdown("### Top Improvement Areas")
            for x in improvements:
                st.write("•",x)

        with c:
            st.markdown("### Frequently Mentioned Terms")
            for x in terms:
                st.write("•",x)

    output=BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary.to_excel(writer,sheet_name="Survey Summary",index=False)
        if not questions.empty:
            questions.to_excel(writer,sheet_name="Question Scores",index=False)
        if not comments.empty:
            comments.to_excel(writer,sheet_name="Comments",index=False)

    st.download_button(
        "Download Analysis Workbook",
        output.getvalue(),
        file_name="feedback_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
