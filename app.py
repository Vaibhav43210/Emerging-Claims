import streamlit as st
import pandas as pd
import requests, os
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
# Read Hugging Face API key from Streamlit secrets or environment variable
api_key = st.secrets.get("HUGGINGFACE_API_KEY") if hasattr(st, "secrets") else None
if not api_key:
    api_key = os.environ.get("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("Missing Hugging Face API key. Set HUGGINGFACE_API_KEY in Streamlit secrets or environment.")
    st.stop()
headers = {"Authorization": f"Bearer {api_key}"}

# Candidate labels
candidate_labels = [
     "Cyber", "Water",
    "Professional Neglicence", "Injured/Illness", "Malicious Damage"
]

# Query function
def query_model(text):
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": candidate_labels}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Streamlit app
st.set_page_config(layout="wide")
st.title("Emerging Risk Intelligence Engine")
tabs = st.tabs(["Data Load", "NLP Inference", "Aggregation", "Emerging Risk Engine", "Visualization", "Insights"])

# Tab 1: Data Load
with tabs[0]:
    uploaded_file = st.file_uploader("Upload Claims Excel File", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="emerging_risk_claims", engine="openpyxl")
        st.session_state.df = df
        st.write("### Preview of Claims Data")
        st.dataframe(df.head(10))

# Tab 2: NLP Inference
with tabs[1]:
    if "df" in st.session_state:
        df = st.session_state.df
        if st.button("Run NLP Classification on First 50 Claims"):
            results = []
            for i, row in df.head(50).iterrows():
                desc = row['Claims_Description']
                claim_id = row['Claim_ID']
                prediction = query_model(desc)
                if 'labels' in prediction:
                    label = prediction['labels'][0]
                    score = prediction['scores'][0]
                else:
                    label = "Error"
                    score = 0.0
                results.append({
                    "Claim_ID": claim_id,
                    "Claims_Description": desc,
                    "Emerging_Risk_Category": label,
                    "Confidence_Score": round(score, 2)
                })
            result_df = pd.DataFrame(results)
            st.session_state.result_df = result_df
            st.write("### NLP Classification Results")
            st.dataframe(result_df)
    else:
        st.warning("Please upload data in Tab 1")

# Tab 3: Aggregation
with tabs[2]:
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df
        agg_df = result_df.groupby("Emerging_Risk_Category").agg({"Claim_ID": "count", "Policy_Number":"count", "Confidence_Score": "mean"}).reset_index()
        agg_df.columns = ["Emerging_Risk_Category", "Claim_Count", "Policy_count", "Avg_Confidence"]
        st.write("### Aggregated Risk Summary")
        st.dataframe(agg_df)
    else:
        st.warning("Run NLP Inference in Tab 2")

# Tab 4: Emerging Risk Engine
with tabs[3]:
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df
        top_risks = result_df['Emerging_Risk_Category'].value_counts().head(5)
        st.write("### Top Emerging Risks")
        st.bar_chart(top_risks)
        new_risks = result_df[result_df['Confidence_Score'] < 0.5]
        st.write("### Low Confidence Predictions (Potential New Risks)")
        st.dataframe(new_risks)
    else:
        st.warning("Run NLP Inference in Tab 2")

# Tab 5: Visualization
with tabs[4]:
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df
        fig = px.histogram(result_df, x="Emerging_Risk_Category", title="Distribution of Risk Categories")
        st.plotly_chart(fig)

        text = " ".join(result_df['Claims_Description'].dropna().tolist())
        wc = WordCloud(width=800, height=400).generate(text)
        st.write("### Word Cloud of Claims Descriptions")
        fig_wc, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    else:
        st.warning("Run NLP Inference in Tab 2")

# Tab 6: Insights
with tabs[5]:
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df
        st.write("### Key Insights")
        st.markdown("- Most frequent risk category: **{}**".format(result_df['Emerging_Risk_Category'].mode()[0]))
        st.markdown("- Average confidence score: **{:.2f}**".format(result_df['Confidence_Score'].mean()))
        st.markdown("- Claims with low confidence (< 0.5): **{}**".format((result_df['Confidence_Score'] < 0.5).sum()))
        st.download_button("Download Full Results", result_df.to_csv(index=False), "emerging_risk_results.csv")
    else:
        st.warning("Run NLP Inference in Tab 2")