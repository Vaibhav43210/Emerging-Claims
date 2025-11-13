import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# -------------------------------
# Helper Functions
# -------------------------------
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
    return f"{int(n)}"

# -------------------------------
# Load CSV and initialize session state
# -------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv('emerging_claims_risk_20Krecords.csv')  # Replace with actual path

df = st.session_state.df

# Normalize columns
df['LOB'] = df['LOB'].astype(str).str.strip()
df['Risk_Category'] = df['Risk_Category'].astype(str).str.strip()

df = df[df['Loss Year']!=2026]

# -------------------------------
# Sidebar Filters (Collapsible)
# -------------------------------

# --- Custom CSS for Sidebar ---
st.markdown("""
    <style>
        /* Make sidebar collapsible and adjust width */
        [data-testid="stSidebar"] {
            width: 100px; /* Adjust sidebar width */
        }

        /* Hide sidebar by default and show toggle button */
        [data-testid="stSidebar"] {
            transform: translateX(-100px);
            transition: transform 0.3s ease-in-out;
        }
        [data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0);
        }

        /* Style sidebar content */
        [data-testid="stSidebar"] .css-1d391kg {
            padding: 10px;
            font-size: 14px;
        }

        /* Style multiselect labels */
        .stMultiSelect label {
            font-size: 14px;
            font-weight: 600;
            color: #333;
        }

        /* Add spacing between widgets */
        .stMultiSelect, .stSelectbox {
            margin-bottom: 15px;
        }

        /* Style expander header */
        .streamlit-expanderHeader {
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Filters ---
st.sidebar.header("Global Filters")
with st.sidebar.expander("Global Filters", expanded=False):
    lob_options = sorted(df['LOB'].dropna().unique().tolist())
    risk_options = sorted(df['Risk_Category'].dropna().unique().tolist())
    year_options = sorted(df['Loss Year'].dropna().unique().tolist())

    selected_lob = st.multiselect("Select Line of Business", lob_options, default=lob_options)
    selected_risk = st.multiselect("Select Risk Category", risk_options, default=risk_options)
    selected_year = st.multiselect("Select Loss Year", year_options, default=year_options)


print('Preeti needs LOB',selected_lob)
print('Preeti needs Risk',selected_risk)
print('Preeti needs Years',selected_year)

# Apply Filters
filtered_df = df[
    df['LOB'].isin(selected_lob) 
    &
    df['Risk_Category'].isin(selected_risk) 
    &
    df['Loss Year'].isin(selected_year)
]

# -------------------------------
# Tabs
# -------------------------------
st.title("Emerging Risk Intelligence Engine")
tab1, tab2, tab3 = st.tabs(["Risk Category Classification", "Risk Category Analysis YOY", "Sub-Category Deep Dive"])

# -------------------------------
# Tab 1: DataFrame + Word Cloud
# -------------------------------
with tab1:
    st.header("Risk Category Classification")
    if filtered_df.empty:
        st.warning("No data available for selected filters.")

    else:
        display_df = filtered_df.rename(columns={
            'Claim_ID': 'Claim Identifier',
            'Claims_Description': 'Claim Comments',
            'LOB': 'Line of Business',
            'Loss_Date': 'Event Date',
            'Risk_Category': 'Risk Category',
            'Sub_Risk_Category': 'Sub Risk Category'
        })[['Claim Identifier', 'Claim Comments', 'Line of Business', 'Event Date', 'Risk Category', 'Sub Risk Category']]

        st.subheader("Claims Data")
        styled_df = display_df.style.hide(axis="index")
        st.dataframe(styled_df, use_container_width=True, height=300)

        st.subheader("Word Cloud for Claim Descriptions")
        text = ' '.join(filtered_df['Claims_Description'].dropna().astype(str))
        if text.strip():
            wordcloud = WordCloud(width=1200, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.warning("No descriptions available for selected filters.")

# -------------------------------
# Tab 2: KPI Tiles (Smaller, Single Row)
# -------------------------------
with tab2:
    st.header("Key Performance Indicators")

    df2 = df[
    df['LOB'].isin(selected_lob) 
    &
    df['Risk_Category'].isin(selected_risk) 
    &
    df['Loss Year'].isin(selected_year)
]

    # st.info(f"Showing {len(filtered_df):,} records out of {len(df):,}")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:
        print('Preeti needs d',df2.shape[0])
        total_claims = df2['Claim_ID'].nunique()
        total_settled = filtered_df['Final_Settled_Amount'].sum()
        total_recovery = filtered_df['Recovery_Amount'].sum()
        total_net_loss = filtered_df['Net_Loss'].sum()
        total_loss_ratio = total_net_loss / filtered_df['Insured_Value'].sum()

        # st.write("### Key KPIs")
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; gap:8px; flex-wrap:wrap;">
            <div style="flex:1; background:#f8f9fa; padding:10px; border-radius:8px; text-align:center; box-shadow:1px 1px 4px rgba(0,0,0,0.1);">
                <h5>📄 Total Claims</h5>
                <p style="font-size:18px; color:#0073e6;">{human_count(total_claims)}</p>
            </div>
            <div style="flex:1; background:#f8f9fa; padding:10px; border-radius:8px; text-align:center; box-shadow:1px 1px 4px rgba(0,0,0,0.1);">
                <h5>💰 Settled Amount</h5>
                <p style="font-size:18px; color:#0073e6;">{human_currency(total_settled)}</p>
            </div>
            <div style="flex:1; background:#f8f9fa; padding:10px; border-radius:8px; text-align:center; box-shadow:1px 1px 4px rgba(0,0,0,0.1);">
                <h5>🔄 Recovery Amount</h5>
                <p style="font-size:18px; color:#0073e6;">{human_currency(total_recovery)}</p>
            </div>
            <div style="flex:1; background:#f8f9fa; padding:10px; border-radius:8px; text-align:center; box-shadow:1px 1px 4px rgba(0,0,0,0.1);">
                <h5>⚠️ Net Loss</h5>
                <p style="font-size:18px; color:#0073e6;">{human_currency(total_net_loss)}</p>
            </div>
            <div style="flex:1; background:#f8f9fa; padding:10px; border-radius:8px; text-align:center; box-shadow:1px 1px 4px rgba(0,0,0,0.1);">
                <h5>📊 Loss Ratio</h5>
                <p style="font-size:18px; color:#0073e6;">{total_loss_ratio:.2f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


        st.header("Year-over-Year Emerging Risk Analysis")

        # YOY Summary
        yoy_summary = df2.groupby(['Loss Year', 'Risk_Category']).agg({
            'Claim_ID': 'count',
            'Net_Loss': 'sum'
        }).reset_index().rename(columns={'Claim_ID': 'Claim_Count', 'Net_Loss': 'Total_Net_Loss'})

    

        
        # Sort and calculate YOY growth
        yoy_summary = yoy_summary.sort_values(['Risk_Category', 'Loss Year'])
        yoy_summary['Claim_Growth'] = yoy_summary.groupby('Risk_Category')['Claim_Count'].pct_change()
        yoy_summary['Loss_Growth'] = yoy_summary.groupby('Risk_Category')['Total_Net_Loss'].pct_change()

        # Generate bullet lists
        bullet_claims = []
        bullet_loss = []
        years = sorted(yoy_summary['Loss Year'].unique())

        
        # Function to assign color tags based on growth
        def growth_tag(value):
            if value >= 0.3:  # High growth
                return f"<span style='color:red;font-weight:bold;'>{value:.2%}</span>"
            elif value >= 0.15:  # Moderate growth
                return f"<span style='color:orange;font-weight:bold;'>{value:.2%}</span>"
            else:  # Low growth
                return f"<span style='color:green;font-weight:bold;'>{value:.2%}</span>"



        for i in range(len(years) - 1):
            start_year, end_year = years[i], years[i+1]
            period_data = yoy_summary[yoy_summary['Loss Year'] == end_year]

            # Highest claim growth category
            max_claim_row = period_data.sort_values('Claim_Growth', ascending=False).iloc[0]
            bullet_claims.append(f"- {start_year}-{end_year}: {max_claim_row['Risk_Category']} ({growth_tag(max_claim_row['Claim_Growth'])})")

            # Highest net loss growth category
            max_loss_row = period_data.sort_values('Loss_Growth', ascending=False).iloc[0]
            bullet_loss.append(f"- {start_year}-{end_year}: {max_loss_row['Risk_Category']} ({growth_tag(max_loss_row['Loss_Growth'])})")

        # Overall insights
        max_growth_category = yoy_summary.loc[yoy_summary['Claim_Growth'].idxmax(), 'Risk_Category']
        max_growth_value = yoy_summary['Claim_Growth'].max()
        claim_insight = f"**Overall Insight:** {max_growth_category} shows the highest YOY growth in claim count overall ({max_growth_value:.2%})."
        claim_action = f"**Actionable:** Review underwriting criteria and pricing for {max_growth_category}."

        loss_volatility = yoy_summary.groupby('Risk_Category')['Total_Net_Loss'].std() / yoy_summary.groupby('Risk_Category')['Total_Net_Loss'].mean()
        volatile_category = loss_volatility.idxmax()
        volatility_value = loss_volatility.max()
        loss_insight = f"**Overall Insight:** {volatile_category} has the highest volatility in net loss ({volatility_value:.2f})."
        loss_action = f"**Actionable:** Consider reinsurance strategies and reserve adjustments for {volatile_category}."


        # Create two columns for side-by-side charts
        # col1, col2 = st.columns([1.5,1.5])

    #     with col1:

    # Custom color palette for Risk Categories
        color_map = {
            'Injury/Illness': '#1f77b4',           # Blue
            'Malicious Activity': '#ff7f0e',       # Orange
            'Natural Disaster': '#d62728',         # Red
            'Operational Error': '#9467bd',        # Purple
            'Product/Service Failure': '#2ca02c'   # Green
        }

    #        
        
        # Create subplots: 1 row, 2 columns
        fig = make_subplots(rows=1, cols=2, subplot_titles=("YOY Claim Count", "YOY Net Loss"),
                            column_widths=[0.4, 0.6],horizontal_spacing=0.20)

        # Add Claim Count traces
        for category in yoy_summary['Risk_Category'].unique():
            data = yoy_summary[yoy_summary['Risk_Category'] == category]
            fig.add_trace(
                go.Scatter(x=data['Loss Year'], y=data['Claim_Count'], mode='lines+markers', name=category,
                line=dict(color=color_map[category])
    ),
                row=1, col=1
            )

        # Add Net Loss traces (no legend to avoid duplication)
        for category in yoy_summary['Risk_Category'].unique():
            data = yoy_summary[yoy_summary['Risk_Category'] == category]
            fig.add_trace(
                go.Scatter(x=data['Loss Year'], y=data['Total_Net_Loss'], mode='lines+markers', showlegend=False,
                line=dict(color=color_map[category])
            ),
                row=1, col=2
            )

        
        # Layout adjustments
        fig.update_layout(
            title_text="Year-over-Year Trends",
            title_font=dict(size=18, color='darkblue'),
            legend=dict(x=1.05, y=1, bordercolor="LightGray", borderwidth=1, font=dict(size=12)),
            margin=dict(l=40, r=150, t=80, b=40),
            # xaxis=dict(showgrid=True),
            # yaxis=dict(showgrid=True),
            height=500
    )
        
        # Update axis titles for each subplot
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Number of Claims", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Total Net Loss (GBP)", row=1, col=2)



        st.plotly_chart(fig, use_container_width=True)

        
        # --- Display Insights in Two Columns with Styling ---
        col1, col2 = st.columns(2)
            
        with col1:
            
            st.markdown("### Claim Count Insights")
            st.markdown(claim_insight)
            st.markdown(claim_action)
            st.markdown("**YOY Growth in Claims:**", unsafe_allow_html=True)
            st.markdown("<br>".join(bullet_claims), unsafe_allow_html=True)
        
        with col2:
            
            st.markdown("### Net Loss Insights")
            st.markdown(loss_insight)
            st.markdown(loss_action)
            st.markdown("**YOY Growth in Net Loss:**", unsafe_allow_html=True)
            st.markdown("<br>".join(bullet_loss), unsafe_allow_html=True)



# --- Third Tab ---


with tab3:
    st.header("Sub-Category Breakdown: Loss Ratio Trends")

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
    else:
        # Convert numeric columns
        filtered_df['Insured_Value'] = pd.to_numeric(filtered_df['Insured_Value'], errors='coerce')
        filtered_df['Net_Loss'] = pd.to_numeric(filtered_df['Net_Loss'], errors='coerce')

        # Aggregate data
        agg_df = filtered_df.groupby(['Risk_Category', 'Sub_Risk_Category', 'Loss Year']).agg({
            'Net_Loss': 'sum',
            'Insured_Value': 'sum'
        }).reset_index()

        if agg_df.empty:
            st.warning("No sub-category data available for selected filters.")
        else:
            # Calculate Loss Ratio
            agg_df['Loss_Ratio'] = agg_df['Net_Loss'] / agg_df['Insured_Value']

            # Global color map for consistency
            unique_subs = agg_df['Sub_Risk_Category'].unique()
            color_palette = px.colors.qualitative.Set2
            sub_colors = {sub: color_palette[i % len(color_palette)] for i, sub in enumerate(unique_subs)}

            # Loop through each Risk Category
            for category in agg_df['Risk_Category'].unique():
                cat_data = agg_df[agg_df['Risk_Category'] == category]

                # Sort subcategories by max Loss Ratio
                avg_loss_ratio = cat_data.groupby('Sub_Risk_Category')['Loss_Ratio'].max().sort_values(ascending=False)
                avg_loss_ratio_df = avg_loss_ratio.reset_index()

                # Extract subcategories for this Risk Category
                sub_categories = avg_loss_ratio_df['Sub_Risk_Category'].unique()

                # Create figure for Loss Ratio only
                fig = go.Figure()

                for sub_cat in sub_categories:
                    sub_data = cat_data[cat_data['Sub_Risk_Category'] == sub_cat]
                    fig.add_trace(go.Scatter(
                        x=sub_data['Loss Year'], y=sub_data['Loss_Ratio'],
                        mode='lines+markers', name=sub_cat,
                        line=dict(color=sub_colors[sub_cat])
                    ))

                # Layout adjustments
                fig.update_layout(
                    title_text=f"{category}: Loss Ratio by Sub-Category",
                    title_x=0.5,  # Center align title
                    height=400 + len(sub_categories)*50,  # Dynamic height
                    margin=dict(t=80, b=80),
                    legend=dict(x=1.05, y=1, orientation='v')  # Right aligned legend
                )

                fig.update_xaxes(title_text="Year")
                fig.update_yaxes(title_text="Loss Ratio")

                # Display chart
                st.plotly_chart(fig, use_container_width=True)

                # --- Insights Section ---
                st.markdown(f"### Insights for {category}")
                top_loss_ratio_subs = avg_loss_ratio.head(2)
                for sub, val in top_loss_ratio_subs.items():
                    st.markdown(f"- **{sub}**: Max Loss Ratio {val:.2f}")

                # --- Actionable Recommendations ---
                st.markdown(f"### Actionable Recommendations for {category}")
                for sub_cat in top_loss_ratio_subs.index:
                    lr = avg_loss_ratio[sub_cat]
                    if lr >= 0.20:
                        st.markdown(f"- **{sub_cat}**: <span style='color:red;font-weight:bold;'>High Risk</span> → Stricter underwriting and risk mitigation.", unsafe_allow_html=True)
                    elif 0.15 <= lr < 0.20:
                        st.markdown(f"- **{sub_cat}**: <span style='color:orange;font-weight:bold;'>Moderate Risk</span> → Review pricing and coverage terms.", unsafe_allow_html=True)
                    else:
                        st.markdown(f"- **{sub_cat}**: <span style='color:green;font-weight:bold;'>Low Risk</span> → Maintain current strategy, monitor trends.", unsafe_allow_html=True)



