import streamlit as st
import pandas as pd
import requests, os, time
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from news_feed import render_news_feed_insights
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(layout="wide")

# Sidebar: Model selection
model_options = {
    "Fast (DistilBART)": "valhalla/distilbart-mnli-12-3", 
    "Accurate (DeBERTa)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "Multilingual (XLM-RoBERTa)": "joeddav/xlm-roberta-large-xnli"

}
# selected_model = st.sidebar.selectbox("Select NLP Model", list(model_options.keys()))
# API_URL = f"https://router.huggingface.co/hf-inference/models/{model_options[selected_model]}"
# API_URL = f"https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
API_URL = f"https://router.huggingface.co/hf-inference/models/valhalla/distilbart-mnli-12-3"

# Hugging Face API key
api_key = st.secrets.get("HUGGINGFACE_API_KEY") if hasattr(st, "secrets") else os.environ.get("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("Missing Hugging Face API key. Set HUGGINGFACE_API_KEY in Streamlit secrets or environment.")
    st.stop()
headers = {"Authorization": f"Bearer {api_key}"}

# Candidate labels
# candidate_labels = ["Cyber", "Natural Disaster", "Professional Negligence", "Injured/Illness", "Malicious Damage", "Fire"]
candidate_labels = ['Operational Error',  'Malicious Activity', 'Natural Disaster', 'Product/Service Failure' , 'Injury/Illness']

# Human-readable currency/count formatter
def human_currency(n):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "N/A"
    try:
        n = float(n)
    except Exception:
        return str(n)
    abs_n = abs(n)
    if abs_n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if abs_n >= 1_000:
        return f"${n/1_000:.2f}K"
    return f"${n:,.2f}"

def human_count(n):
    if n is None:
        return "0"
    try:
        n = float(n)
    except Exception:
        return str(n)
    abs_n = abs(n)
    if abs_n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs_n >= 1_000:
        return f"{n/1_000:.2f}K"
    # integer if small
    return f"{int(n)}"


# Query function
def query_model(text, retries=2, timeout=90):
    # candidate_labels = generate_candidate_labels(text)
    # print(f"Generated candidate labels: {candidate_labels}")
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    for attempt in range(retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                print(f"Request failed after {retries} retries: {e}")
                return {"error": str(e)}
            time.sleep(2)
            continue
        if not (200 <= response.status_code < 300):
            if 500 <= response.status_code < 600 and attempt < retries:
                time.sleep(2)
                continue
            print(f"Model API error {response.status_code}: {response.text.strip()[:1000]}")
            return {"error": f"HTTP {response.status_code}: {response.text.strip()[:1000]}"}
        try:
            return response.json()
        except ValueError:
            print(f"Invalid JSON response: {response.text.strip()[:1000]}")
            return {"error": f"Invalid JSON response: {response.text.strip()[:1000]}"}
    return {"error": "Unknown error contacting model"}

# Streamlit app setup

st.title("Emerging Risk Intelligence Engine")
tabs = st.tabs(["Data Load", "Risk Category", "Risk Analysis", "Dimensional YOY Comparison", "Summarized Newsletter", "Emerging News Feed"])

# Tab 1: Data Load
with tabs[0]:
    uploaded_file = st.file_uploader("Upload Claims Excel File", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, sheet_name="emerging_risk_claims", engine="openpyxl")
        st.session_state.df = df
        st.write("### Preview of Claims Data")
        st.dataframe(df.head(10))

# Tab 2: NLP Inference
# with tabs[1]:
#     if "df" in st.session_state:
#         df = st.session_state.df
#         num_claims = st.slider("Select number of claims to process", min_value=1, max_value=100, value=50, step=5)
#         if st.button("Get Emerging Risk Categories"):
#             with st.spinner("Getting Categories..."):
#                 results = []
#                 progress = st.progress(0)
#                 for i, row in df.head(num_claims).iterrows():
#                     desc = row['Claims_Description']
#                     claim_id = row['Claim_ID']
#                     prediction = query_model(desc)
#                     print(f"Processed Claim ID {claim_id}: {prediction}")
#                     if isinstance(prediction, dict) and prediction.get("error"):
#                         label = "Error"
#                         score = 0.0
#                         print(f"Model error for Claim {claim_id}: {prediction['error']}")
#                     elif isinstance(prediction, list) and len(prediction) > 0:
#                         label = prediction[0]['label']
#                         score = prediction[0]['score']

#                         print(f"Claim ID {claim_id} classified as {label} with score {score}")
#                     else:
#                         label = "Error"
#                         score = 0.0
#                         print(f"Unexpected model response for Claim ID {claim_id}: {prediction}")
#                     results.append({
#                         "Claim_ID": claim_id,
#                         "Claims_Description": desc,
#                         "Emerging_Risk_Category": label,
#                         "Confidence_Score": round(score, 2)
#                     })
#                     progress.progress((i + 1) / num_claims)
#                 result_df = pd.DataFrame(results)
#                 st.session_state.result_df = result_df
#                 st.write("### NLP Classification Results")
#                 st.dataframe(result_df)
#     else:
#         st.warning("Please upload data in Tab 1")


@st.cache_data(show_spinner=False)
def cached_query_model(description):
    return query_model(description)


def process_claim(row):
    desc = row['Claims_Description']
    claim_id = row['Claim_ID']
    prediction = cached_query_model(desc)

    if isinstance(prediction, dict) and prediction.get("error"):
        label = "Error"
        score = 0.0
    elif isinstance(prediction, list) and len(prediction) > 0:
        label = prediction[0]['label']
        score = prediction[0]['score']
    else:
        label = "Error"
        score = 0.0

    return {
        "Claim_ID": claim_id,
        "Claims_Description": desc,
        "Emerging_Risk_Category": label,
        "Confidence_Score": round(score, 2)
    }

with tabs[1]:
    if "df" in st.session_state:
        df = st.session_state.df
        df_subset = df.head(1000)  # Select only first 50 claims

        if st.button("Get Emerging Risk Categories for 1000 Claims only"):
            with st.spinner("Processing 1000 claims..."):
                total_claims = len(df)
                rows = [row for _, row in df_subset.iterrows()]
                results = []

                progress_placeholder = st.empty()
                progress_bar = progress_placeholder.progress(0)

                # Parallel + Cached
                with ThreadPoolExecutor(max_workers=100) as executor:
                    futures = {executor.submit(process_claim, row): idx for idx, row in enumerate(rows)}
                    for i, future in enumerate(as_completed(futures)):
                        result = future.result()
                        results.append(result)
                        progress_bar.progress((i + 1) / len(rows))

                result_df = pd.DataFrame(results)
                
                # Sort the result_df by Confidence_Score in descending order
                sorted_df = result_df.sort_values(by="Confidence_Score", ascending=False)

                st.session_state.result_df = sorted_df

                st.write("### Emerging Categories Results")
                st.dataframe(sorted_df.drop_duplicates(subset=['Claims_Description']).head(10))

                st.download_button(
                    label="Download Top 1000 Results",
                    data=sorted_df.to_csv(index=False),
                    file_name="1000claimsonly.csv",
                    mime="text/csv"
                )
        if st.button("Get Emerging Risk Categories for All Claims"):
                with st.spinner("Processing all claims..."):
                    total_claims = len(df)
                    rows = [row for _, row in df.iterrows()]
                    results = []

                    progress_placeholder = st.empty()
                    progress_bar = progress_placeholder.progress(0)

                    # Parallel + Cached
                    with ThreadPoolExecutor(max_workers=100) as executor:
                        futures = {executor.submit(process_claim, row): idx for idx, row in enumerate(rows)}
                        for i, future in enumerate(as_completed(futures)):
                            result = future.result()
                            results.append(result)
                            progress_bar.progress((i + 1) / len(rows))

                    result_df = pd.DataFrame(results)
                    # Sort the result_df by Confidence_Score in descending order
                    sorted_df = result_df.sort_values(by="Confidence_Score", ascending=False)

                    st.session_state.result_df = sorted_df
                    # st.session_state.result_df = result_df

                    st.write("### Emerging Categories Results")
                    st.dataframe(sorted_df.drop_duplicates(subset=['Claims_Description']).head(10))

                    st.download_button(
                        label="Download All Results",
                        data=sorted_df.to_csv(index=False),
                        file_name="All_claims.csv",
                        mime="text/csv"
                    )
    else:
        st.warning("Please upload data in Data Upload Tab")




# Tab 3: Risk Analysis (Merged)
with tabs[2]:
    if "df" in st.session_state:
        df = st.session_state.df

        # If NLP results exist from Tab 2, merge Emerging Risk Category & Confidence
        if "result_df" in st.session_state:
            res_df = st.session_state.result_df[['Claim_ID', 'Emerging_Risk_Category', 'Confidence_Score']].drop_duplicates(subset=['Claim_ID'])
            df = df.merge(res_df, on='Claim_ID', how='left')
            # Persist merged results back to session_state so other tabs (Visualization/Insights) see them
            st.session_state.df = df
        else:
            # Ensure column exists to avoid KeyErrors later
            if 'Emerging_Risk_Category' not in df.columns:
                df['Emerging_Risk_Category'] = None
            if 'Confidence_Score' not in df.columns:
                df['Confidence_Score'] = None
            st.session_state.df = df

        # Convert dates and add Year
        df['Claim_Date'] = pd.to_datetime(df['Claim_Date'], errors='coerce')
        df['Year'] = df['Claim_Date'].dt.year

        st.subheader("Interactive Risk Analysis")

        # Filters (includes Emerging Risk Category)
        years = sorted(df['Year'].dropna().unique())
        # product_lines = sorted(df['Product_Line'].dropna().unique())
        # industries = sorted(df['Industry_Segment'].dropna().unique())
        # countries = sorted(df['Country'].dropna().unique())
        # claim_types = sorted(df['Claim_Type'].dropna().unique())
        categories = sorted(df['Emerging_Risk_Category'].dropna().unique())

        selected_years = st.multiselect("Select Years", years, default=years)
        # selected_products = st.multiselect("Select Product Lines", product_lines, default=product_lines)
        # selected_industries = st.multiselect("Select Industry Segments", industries, default=industries)
        # selected_countries = st.multiselect("Select Countries", countries, default=countries)
        # selected_claim_types = st.multiselect("Select Claim Types", claim_types, default=claim_types)
        selected_categories = st.multiselect("Select Emerging Risk Categories", categories, default=categories if categories else [])

        # Apply filters (handle empty category list)
        filtered_df = df[
            (df['Year'].isin(selected_years)) 
            # &
            # (df['Product_Line'].isin(selected_products)) &
            # (df['Industry_Segment'].isin(selected_industries)) &
            # (df['Country'].isin(selected_countries)) &
            # (df['Claim_Type'].isin(selected_claim_types))
        ]
        if selected_categories:
            filtered_df = filtered_df[filtered_df['Emerging_Risk_Category'].isin(selected_categories)]

        # KPI Cards
        total_claims = filtered_df['Claim_ID'].nunique()
        total_reported_loss = filtered_df['Reported_Loss_Amount'].sum()
        total_settled = filtered_df['Final_Settled_Amount'].sum()
        total_recovery = filtered_df['Recovery_Amount'].sum()
        avg_loss_ratio = filtered_df['Loss_Ratio'].mean()

        st.write("### Key KPIs")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Claims", human_count(total_claims))
        col2.metric("Reported Loss", human_currency(total_reported_loss))
        col3.metric("Settled Amount", human_currency(total_settled))
        col4.metric("Recovery Amount", human_currency(total_recovery))
        col5.metric("Avg Loss Ratio", f"{avg_loss_ratio:.2%}" if not pd.isna(avg_loss_ratio) else "N/A")

        # Aggregated Summary: yearly totals and Year x Emerging Risk Category breakdown
        yearly_agg = filtered_df.groupby("Year").agg({
            "Claim_ID": "count",
            "Reported_Loss_Amount": "sum",
            "Final_Settled_Amount": "sum",
            "Recovery_Amount": "sum",
            "Net_Loss": "sum",
            "Loss_Ratio": "mean"
        }).reset_index()
        yearly_agg.columns = ["Year", "Claim_Count", "Total_Reported_Loss", "Total_Settled", "Total_Recovery", "Total_Net_Loss", "Avg_Loss_Ratio"]

        category_agg = filtered_df.groupby(["Year", "Emerging_Risk_Category"]).agg({
            "Claim_ID": "count",
            "Reported_Loss_Amount": "sum",
            "Final_Settled_Amount": "sum"
        }).reset_index().rename(columns={"Claim_ID": "Claim_Count"})

        st.write("### Aggregated Summary - Yearly Totals")
        st.dataframe(yearly_agg)
        st.download_button("Download Yearly Aggregated Summary", yearly_agg.to_csv(index=False), "yearly_aggregated_summary.csv")

        st.write("### Aggregated Summary - Year x Emerging Risk Category")
        st.dataframe(category_agg)
        st.download_button("Download Category Aggregated Summary", category_agg.to_csv(index=False), "category_aggregated_summary.csv")

        # Charts
        st.write("### Interactive Charts")
        fig_claim = px.bar(yearly_agg, x="Year", y="Total_Settled", title="Claim Volume Over Years")
        fig_claim.update_yaxes(tickformat=".2s", tickprefix="$")
        st.plotly_chart(fig_claim, use_container_width=True)

        fig_recovery = px.line(yearly_agg, x="Year", y="Total_Recovery", title="Recovery Amount Over Years", markers=True)
        fig_recovery.update_yaxes(tickformat=".2s", tickprefix="$")
        st.plotly_chart(fig_recovery, use_container_width=True)

        fig_loss = px.line(yearly_agg, x="Year", y="Total_Reported_Loss", title="Reported Loss Amount Over Years", markers=True)
        fig_loss.update_yaxes(tickformat=".2s", tickprefix="$")
        st.plotly_chart(fig_loss, use_container_width=True)

        fig_ratio = px.line(yearly_agg, x="Year", y="Avg_Loss_Ratio", title="Average Loss Ratio Over Years", markers=True)
        fig_ratio.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_ratio, use_container_width=True)

        
        # Chart: Year-on-Year by Emerging Risk Category
        if not category_agg.empty:
            fig_cat = px.bar(category_agg, x="Year", y="Claim_Count", color="Emerging_Risk_Category",
                             title="Year-on-Year Claim Count by Emerging Risk Category", barmode='group')
            fig_cat.update_yaxes(tickformat=".2s")
            st.plotly_chart(fig_cat, use_container_width=True)

    else:
        st.warning("Please upload data in Data Tab")

# # Tab 4: Visualization
# with tabs[3]:
#     if "df" in st.session_state:
#         df = st.session_state.df
#         st.subheader("WordClouds & Year-on-Year Comparison")

#         # Only show WordCloud UI when there is at least one non-null category
#         if 'Claims_Description' in df.columns and 'Emerging_Risk_Category' in df.columns:
#             categories = df['Emerging_Risk_Category'].dropna().unique().tolist()
#             if categories:
#                 selected_category = st.selectbox("Select Category for WordCloud", categories)
#                 text = " ".join(df[df['Emerging_Risk_Category'] == selected_category]['Claims_Description'].dropna().astype(str))
#                 if text.strip():
#                     try:
#                         wc = WordCloud(width=800, height=400, background_color='white').generate(text)
#                         fig_wc, ax = plt.subplots()
#                         ax.imshow(wc, interpolation='bilinear')
#                         ax.axis("off")
#                         st.pyplot(fig_wc)
#                     except ValueError:
#                         st.info("Not enough text to generate a word cloud for the selected category.")
#                 else:
#                     st.info("No claim descriptions available for the selected category to generate a word cloud.")
#             else:
#                 st.info("No Emerging Risk Categories available. Run NLP Inference in Tab 2 first.")

#         # Year-on-year: fill missing categories with 'Uncategorized' to ensure grouping works
#         yoy_df = df.copy()
#         # if 'Emerging_Risk_Category' not in yoy_df.columns:
#         #     yoy_df['Emerging_Risk_Category'] = 'Uncategorized'
#         # else:
#         #     yoy_df['Emerging_Risk_Category'] = yoy_df['Emerging_Risk_Category'].fillna('Uncategorized')
#         if 'Year' not in yoy_df.columns:
#             yoy_df['Year'] = pd.to_datetime(yoy_df.get('Claim_Date', pd.Series())).dt.year
#         yoy_df = yoy_df.groupby(['Year', 'Emerging_Risk_Category']).agg({'Claim_ID': 'count'}).reset_index()
#         fig_yoy = px.bar(yoy_df, x='Year', y='Claim_ID', color='Emerging_Risk_Category',
#                                title='Year-on-Year Claim Count by Emerging Risk Category', barmode='group')
#         fig_yoy.update_yaxes(tickformat=".2s")
#         st.plotly_chart(fig_yoy, use_container_width=True)

#         # Sunburst Chart
#         # viz_df = df.copy()    
#         # viz_df['Year'] = pd.to_datetime(viz_df.get('Claim_Date', pd.Series()), errors='coerce').dt.year
#         # print(viz_df)
#         # # # Group data
#         # sunburst_df = viz_df.groupby(['Year', 'Emerging_Risk_Category']).size().reset_index(name='Count')

#         # # # Validate before plotting
#         # if not sunburst_df.empty and sunburst_df['Year'].notna().any() and sunburst_df['Emerging_Risk_Category'].notna().any():
#         #     fig_sunburst = px.sunburst(
#         #         sunburst_df,
#         #         path=['Year', 'Emerging_Risk_Category'],
#         #         values='Count',
#         #         title='Risk Category Distribution by Year'
#         #     )
#         #     st.plotly_chart(fig_sunburst, use_container_width=True)
#         # else:
#         #     st.warning("No valid data available for Sunburst chart.")


#         # # Treemap
#         # treemap_df = viz_df.groupby(['Year', 'Emerging_Risk_Category']).agg({'Final_Settled_Amount': 'sum'}).reset_index()
#         # fig_treemap = px.treemap(treemap_df, path=['Year', 'Emerging_Risk_Category'], values='Final_Settled_Amount',
#         #                          title='Settled Amount by Risk Category and Year')
#         # st.plotly_chart(fig_treemap, use_container_width=True)

#         # # Animated Bar Chart
#         # animated_df = viz_df.groupby(['Year', 'Emerging_Risk_Category']).size().reset_index(name='Claim_Count')
#         # fig_animated = px.bar(animated_df, x='Emerging_Risk_Category', y='Claim_Count', color='Emerging_Risk_Category',
#         #                       animation_frame='Year', title='YOY Claim Count by Risk Category')
#         # st.plotly_chart(fig_animated, use_container_width=True)
#     else:
#         st.warning("Please upload data in Data Tab")

# Tab 4: Visualization
with tabs[3]:
    if "df" in st.session_state:
        df = st.session_state.df

        # ✅ Guard clause to wait for NLP inference
        if 'Emerging_Risk_Category' not in df.columns or df['Emerging_Risk_Category'].dropna().empty:
            st.warning("Waiting for NLP inference. Please run it in Tab 2 to view visualizations.")
            st.stop()

        st.subheader("WordClouds & Year-on-Year Comparison")

        # ✅ WordCloud
        if 'Claims_Description' in df.columns:
            categories = df['Emerging_Risk_Category'].dropna().unique().tolist()
            if categories:
                selected_category = st.selectbox("Select Category for WordCloud", categories)
                text = " ".join(df[df['Emerging_Risk_Category'] == selected_category]['Claims_Description'].dropna().astype(str))
                if text.strip():
                    try:
                        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                        fig_wc, ax = plt.subplots()
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_wc)
                    except ValueError:
                        st.info("Not enough text to generate a word cloud for the selected category.")
                else:
                    st.info("No claim descriptions available for the selected category to generate a word cloud.")
            else:
                st.info("No Emerging Risk Categories available. Run NLP Inference in Tab 2 first.")

        # ✅ Year-on-Year Bar Chart
        yoy_df = df.copy()
        if 'Year' not in yoy_df.columns:
            yoy_df['Year'] = pd.to_datetime(yoy_df.get('Claim_Date', pd.Series()), errors='coerce').dt.year
        yoy_df = yoy_df.groupby(['Year', 'Emerging_Risk_Category']).agg({'Claim_ID': 'count'}).reset_index()
        fig_yoy = px.bar(yoy_df, x='Year', y='Claim_ID', color='Emerging_Risk_Category',
                         title='Year-on-Year Claim Count by Emerging Risk Category', barmode='group')
        fig_yoy.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig_yoy, use_container_width=True)

        # ✅ Sunburst Chart
        viz_df = df.copy()
        viz_df['Year'] = pd.to_datetime(viz_df.get('Claim_Date', pd.Series()), errors='coerce').dt.year
        sunburst_df = viz_df.groupby(['Year', 'Emerging_Risk_Category']).size().reset_index(name='Count')
        if not sunburst_df.empty and sunburst_df['Year'].notna().any() and sunburst_df['Emerging_Risk_Category'].notna().any():
            fig_sunburst = px.sunburst(
                sunburst_df,
                path=['Year', 'Emerging_Risk_Category'],
                values='Count',
                title='Risk Category Distribution by Year'
            )
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning("No valid data available for Sunburst chart.")

        # ✅ Treemap
        treemap_df = viz_df.groupby(['Year', 'Emerging_Risk_Category']).agg({'Final_Settled_Amount': 'sum'}).reset_index()
        fig_treemap = px.treemap(treemap_df, path=['Year', 'Emerging_Risk_Category'], values='Final_Settled_Amount',
                                 title='Settled Amount by Risk Category and Year')
        st.plotly_chart(fig_treemap, use_container_width=True)

        # ✅ Animated Bar Chart
        animated_df = df.groupby(['Year', 'Emerging_Risk_Category']).size().reset_index(name='Claim_Count')
        fig_animated = px.bar(animated_df, x='Emerging_Risk_Category', y='Claim_Count', color='Emerging_Risk_Category',
                              animation_frame='Year', title='YOY Claim Count by Risk Category')
        st.plotly_chart(fig_animated, use_container_width=True)
    else:
        st.warning("Please upload data in Tab 1")


# Tab 5: Insights (LLM-enhanced newsletter)
with tabs[4]:
    if "df" in st.session_state:
        df = st.session_state.df.copy()
        st.subheader("LLM-Based Newsletter Summary")
        summary_points = []

        # Ensure Year exists
        if 'Year' not in df.columns:
            df['Claim_Date'] = pd.to_datetime(df.get('Claim_Date', pd.Series()), errors='coerce')
            df['Year'] = df['Claim_Date'].dt.year

        # Top categories (safe handling if empty)
        top_categories = df['Emerging_Risk_Category'].dropna().value_counts().head(5)
        if not top_categories.empty:
            for category, count in top_categories.items():
                avg_loss = df[df['Emerging_Risk_Category'] == category]['Reported_Loss_Amount'].mean()
                avg_loss_text = f"${avg_loss:,.2f}" if not pd.isna(avg_loss) else "N/A"
                summary_points.append(f"- **{category}**: {count} claims, avg reported loss {avg_loss_text}")
        else:
            summary_points.append("- No Emerging Risk Categories detected yet. Run NLP Inference in Tab 2.")

        # Recent year highlights
        recent_year_series = df['Year'].dropna()
        recent_year = None
        if not recent_year_series.empty:
            recent_year = int(recent_year_series.max())
            recent_data = df[df['Year'] == recent_year]
            if not recent_data.empty:
                recent_counts = recent_data['Emerging_Risk_Category'].dropna().value_counts()
                if not recent_counts.empty:
                    top_recent = recent_counts.idxmax()
                    summary_points.append(f"- In {recent_year}, most frequent risk category: **{top_recent}**.")
                else:
                    summary_points.append(f"- In {recent_year}, no categorized claims were available.")
        else:
            summary_points.append("- No claim year information available to generate recent highlights.")

        # Compute simple year-over-year change for top categories (if enough data)
        yoy_lines = []
        if 'Year' in df.columns and not top_categories.empty:
            pivot = df.dropna(subset=['Emerging_Risk_Category']).groupby(['Year', 'Emerging_Risk_Category']).size().unstack(fill_value=0)
            years_sorted = sorted([y for y in pivot.index if pd.notna(y)])
            if len(years_sorted) >= 2:
                y_latest = years_sorted[-1]
                y_prev = years_sorted[-2]
                for category in top_categories.index:
                    latest = int(pivot.loc[y_latest, category]) if category in pivot.columns and y_latest in pivot.index else 0
                    prev = int(pivot.loc[y_prev, category]) if category in pivot.columns and y_prev in pivot.index else 0
                    change = latest - prev
                    pct = f"{(change/prev):.0%}" if prev != 0 else "N/A" if latest == 0 else "∞"
                    yoy_lines.append(f"- {category}: {prev} → {latest} (Δ {change}, {pct}) between {y_prev} and {y_latest}")
        if yoy_lines:
            summary_points.append("- Year-over-Year changes for top categories:")
            summary_points.extend(yoy_lines)

        # Display the short summary points
        st.write("### Emerging Risk Highlights (Derived)")
        for point in summary_points:
            st.markdown(point)

#         # Build prompt for LLM
#         prompt_header = "You are an insurance claims analyst assistant. Given the derived emerging risk points and trends, produce a short newsletter (3-6 bullets) summarizing current trends, near-term (6-12 months) future possibilities, and 3 actionable recommendations for risk owners.\n\n"
#         derived_text = "\n".join(summary_points)
#         prompt = prompt_header + "Derived data:\n" + derived_text + "\n\nNewsletter:\n"

        # st.markdown("### Generate newsletter using LLM")
        # selected_generation_model = st.selectbox("LLM model (Hugging Face)", ["gpt2", "bigscience/bloom", "mistralai/Mistral-7B-Instruct-v0.1"], index=0)
        # max_tokens = st.slider("Max tokens for generation", min_value=64, max_value=512, value=256, step=64)

        # if st.button("Generate Newsletter (LLM)"):
        #     with st.spinner("Generating newsletter..."):
        #         llm_result = generate_newsletter(prompt, model=selected_generation_model, max_new_tokens=max_tokens)
        #         if isinstance(llm_result, dict) and llm_result.get("error"):
        #             st.error(f"LLM error: {llm_result['error']}")
        #         else:
        #             newsletter_text = llm_result.get("text") if isinstance(llm_result, dict) else str(llm_result)
        #             # Show generated newsletter and allow edits
        #             st.write("### Generated Newsletter")
        #             editable = st.text_area("Edit newsletter before download/publish", value=newsletter_text, height=250)
        #             st.download_button("Download Newsletter Summary", editable, "llm_risk_newsletter.txt")
        # else:
        #     st.info("Click 'Generate Newsletter (LLM)' to produce a newsletter using the selected model.")

    else:
        st.warning("Please upload data in Tab 1")

# Tab 6: News Feed Insights
with tabs[5]:
    st.subheader("News Feed Insights")
    render_news_feed_insights()
    # render_event_dashboard(api_key=st.secrets.get("EVENTREGISTRY_API_KEY") if hasattr(st, "secrets") else os.environ.get("EVENTREGISTRY_API_KEY"))
