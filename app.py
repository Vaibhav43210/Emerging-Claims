import streamlit as st
import pandas as pd
import requests, os, time
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Sidebar: Model selection
model_options = {
    "Fast (DistilBART)": "valhalla/distilbart-mnli-12-3", 
    "Accurate (DeBERTa)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "Multilingual (XLM-RoBERTa)": "joeddav/xlm-roberta-large-xnli"

}
selected_model = st.sidebar.selectbox("Select NLP Model", list(model_options.keys()))
API_URL = f"https://api-inference.huggingface.co/models/{model_options[selected_model]}"

# Hugging Face API key
api_key = st.secrets.get("HUGGINGFACE_API_KEY") if hasattr(st, "secrets") else os.environ.get("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("Missing Hugging Face API key. Set HUGGINGFACE_API_KEY in Streamlit secrets or environment.")
    st.stop()
headers = {"Authorization": f"Bearer {api_key}"}

# Candidate labels
candidate_labels = ["Cyber", "Water", "Professional Negligence", "Injured/Illness", "Malicious Damage", "Fire"]

# Query function
def query_model(text, retries=2, timeout=60):
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    for attempt in range(retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                return {"error": str(e)}
            time.sleep(2)
            continue
        if not (200 <= response.status_code < 300):
            if 500 <= response.status_code < 600 and attempt < retries:
                time.sleep(2)
                continue
            return {"error": f"HTTP {response.status_code}: {response.text.strip()[:1000]}"}
        try:
            return response.json()
        except ValueError:
            return {"error": f"Invalid JSON response: {response.text.strip()[:1000]}"}
    return {"error": "Unknown error contacting model"}

# Streamlit app setup

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
        num_claims = st.slider("Select number of claims to process", min_value=5, max_value=100, value=50, step=5)
        if st.button("Run NLP Classification"):
            with st.spinner("Running NLP classification..."):
                results = []
                progress = st.progress(0)
                for i, row in df.head(num_claims).iterrows():
                    desc = row['Claims_Description']
                    claim_id = row['Claim_ID']
                    prediction = query_model(desc)
                    if isinstance(prediction, dict) and prediction.get("error"):
                        label = "Error"
                        score = 0.0
                        st.error(f"Model error for Claim {claim_id}: {prediction['error']}")
                    elif 'labels' in prediction and 'scores' in prediction:
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
                    progress.progress((i + 1) / num_claims)
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
        agg_df = result_df.groupby("Emerging_Risk_Category").agg({"Claim_ID": "count", "Confidence_Score": "mean"}).reset_index()
        agg_df.columns = ["Emerging_Risk_Category", "Claim_Count", "Avg_Confidence"]
        st.write("### Aggregated Risk Summary")
        st.dataframe(agg_df)
        st.download_button("Download Aggregated Summary", agg_df.to_csv(index=False), "aggregated_risks.csv")
    else:
        st.warning("Run NLP Inference in Tab 2")

# Tab 4: Emerging Risk Engine
with tabs[3]:
    if "result_df" in st.session_state:
        result_df = st.session_state.result_df
        top_risks = result_df['Emerging_Risk_Category'].value_counts().head(5)
        st.write("### Top Emerging Risks")
        st.bar_chart(top_risks)
        threshold = st.slider("Confidence threshold for new risks", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        new_risks = result_df[result_df['Confidence_Score'] < threshold]
        st.write(f"### Claims with Confidence < {threshold}")
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