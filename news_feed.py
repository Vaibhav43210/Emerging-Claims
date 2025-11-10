from datetime import datetime, timedelta
import pandas as pd
import requests
import re
import streamlit as st
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems, ReturnInfo, ArticleInfoFlags, SourceInfoFlags
from rapidfuzz import fuzz

# API keys
EVENTREGISTRY_API_KEY = st.secrets.get("EVENTREGISTRY_API_KEY")
HUGGINGFACE_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")

# Default parameters
DEFAULT_KEYWORDS = ['Operational Error', 'Malicious Activity', 'Natural Disaster', 'Product/Service Failure', 'Injury/Illness']
CLAIMS_KEYWORDS = ['insurance', 'loss', 'settlement', 'compensation', 'damage', 'payout','claim']
DEFAULT_MAX_PER_KEYWORD = 3
DEFAULT_START_DATE = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
DEFAULT_END_DATE = datetime.utcnow().strftime("%Y-%m-%d")
REQUEST_KW = {"timeout": 60}

# Hugging Face summarization
def _hf_summarize(text: str, model_name="google/pegasus-xsum", max_length=130, min_length=30) -> str:
    if not text:
        return ""
    if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY.startswith("REPLACE_"):
        raise RuntimeError("HUGGINGFACE_API_KEY not set")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    models_to_try = [model_name, "sshleifer/distilbart-cnn-12-6", "philschmid/bart-large-cnn-samsum"]

    def chunk_text(t: str, limit: int = 2200) -> list:
        if len(t) <= limit:
            return [t]
        sents = re.split(r'(?<=[.!?])\s+', t)
        chunks, current = [], ""
        for s in sents:
            if len(current) + len(s) + 1 <= limit:
                current += s + " "
            else:
                chunks.append(current.strip())
                current = s + " "
        if current:
            chunks.append(current.strip())
        return chunks

    summaries = []
    for p in chunk_text(text):
        last_exc = None
        for attempt in range(2):
            for m in models_to_try:
                url = f"https://router.huggingface.co/hf-inference/models/{m}"
                payload = {"inputs": p, "parameters": {"max_length": max_length, "min_length": min_length}}
                try:
                    resp = requests.post(url, headers=headers, json=payload, **REQUEST_KW)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    last_exc = e
                    continue
                if isinstance(data, list) and data:
                    txt = data[0].get("summary_text") or data[0].get("generated_text") or ""
                    summaries.append(txt)
                elif isinstance(data, dict) and data.get("error"):
                    last_exc = RuntimeError(data.get("error"))
                    continue
                elif isinstance(data, str):
                    summaries.append(data)
                else:
                    summaries.append(str(data))
                last_exc = None
                break
            if last_exc is None:
                break
        if last_exc:
            raise RuntimeError(f"HF summarization failed: {last_exc}")
    return " ".join(s.strip() for s in summaries if s).strip()

# LLM-based extraction for claims info
def _extract_claims_llm(text: str) -> dict:
    if not text:
        return {"claim_amount": None, "location": None, "incident": None, "intensity": None}
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    url = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"
    prompt = f"""
    Extract the following information from the text:
    - Claim amount (numeric or descriptive)
    - Location (city, state, or country)
    - Incident (short description)
    - Intensity (severity level or descriptive phrase)
    
    Return ONLY valid JSON in this format:
    {{"claim_amount": "<amount or None>", "location": "<location or None>", "incident": "<incident or None>", "intensity": "<intensity or None>"}}

    Text: {text}
    """
    payload = {"inputs": prompt, "parameters": {"max_length": 256}}
    try:
        resp = requests.post(url, headers=headers, json=payload, **REQUEST_KW)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            raw = data[0].get("generated_text") or "{}"
        else:
            raw = str(data)
        match = re.search(r"\{.*\}", raw)
        if match:
            import json
            return json.loads(match.group(0))
    except Exception:
        return {"claim_amount": None, "location": None, "incident": None, "intensity": None}
    return {"claim_amount": None, "location": None, "incident": None, "intensity": None}

# News fetcher
@st.cache_data(show_spinner=True, ttl=30*60)
def _fetch_news_df(_keywords: tuple, _start: str, _end: str, _max_per: int) -> pd.DataFrame:
    if not EVENTREGISTRY_API_KEY or EVENTREGISTRY_API_KEY.startswith("REPLACE_"):
        raise RuntimeError("EVENTREGISTRY_API_KEY not set")

    er = EventRegistry(apiKey=EVENTREGISTRY_API_KEY, allowUseOfArchive=False)
    ret_info = ReturnInfo(articleInfo=ArticleInfoFlags(bodyLen=-1, concepts=False, categories=False, authors=True),
                          sourceInfo=SourceInfoFlags())

    rows = []
    
    for kw in list(_keywords):
        
        
        
        q = QueryArticlesIter(
            keywords=kw,
            # QueryItems.AND([kw,QueryItems.OR(CLAIMS_KEYWORDS)]),
            dateStart=_start,
            dateEnd=_end,
            isDuplicateFilter=True,
            lang="eng"
        )

        
# Fetch top N articles sorted by date
        for art in q.execQuery(er, returnInfo=ret_info, sortBy="date", sortByAsc=False, maxItems=_max_per):
            rows.append({
                "date": art.get("dateTime", ""),
                "keyword": kw,
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "source": (art.get("source") or {}).get("title", ""),
                "authors": ", ".join([author.get("name", "") for author in art.get("authors", [])]),
                "body": art.get("body", "")
            })


    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")

    # Apply LLM extraction
    # extracted = df["body"].apply(_extract_claims_llm)
    # extracted_df = pd.DataFrame(extracted.tolist())
    # df = pd.concat([df, extracted_df], axis=1)

    # Group and summarize
    grouped_df = df.groupby("keyword").agg({
        "body": lambda x: " ".join(x),
        # "claim_amount": lambda x: ", ".join(filter(None, x)),
        # "location": lambda x: ", ".join(filter(None, x)),
        # "incident": lambda x: ", ".join(filter(None, x)),
        # "intensity": lambda x: ", ".join(filter(None, x)),
        "date_parsed": "max"
    }).reset_index()
    grouped_df["summary"] = grouped_df["body"].apply(_hf_summarize)

    return grouped_df.sort_values("date_parsed", ascending=False).reset_index(drop=True)


def deduplicate_news(df, threshold=90):
    seen_titles = []
    deduped_rows = []
    for _, row in df.iterrows():
        title = row['body']
        if not any(fuzz.token_set_ratio(title, seen) > threshold for seen in seen_titles):
            seen_titles.append(title)
            deduped_rows.append(row)
    return pd.DataFrame(deduped_rows)

# Combined summary generator
# def generate_combined_category_summary(category: str, claims_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
#     claims_cat = claims_df[claims_df['Emerging_Risk_Category'] == category]
#     news_cat = news_df[news_df['keyword'] == category]
#     claim_count = claims_cat['Claim_ID'].nunique()
#     avg_loss = claims_cat['Reported_Loss_Amount'].mean()
#     avg_loss_text = f"${avg_loss:,.2f}" if pd.notna(avg_loss) else "N/A"
#     top_prof = "N/A"
#     if 'Claim_Type' in claims_cat.columns and not claims_cat['Claim_Type'].dropna().empty:
#         top_prof = claims_cat['Claim_Type'].dropna().value_counts().idxmax()
#     yoy = claims_cat.groupby('Year').size()
#     years = sorted(yoy.index)
#     yoy_text = ""
#     if len(years) >= 2:
#         prev, latest = yoy.iloc[-2], yoy.iloc[-1]
#         change = latest - prev
#         pct = f"{(change/prev):.0%}" if prev != 0 else "∞"
#         yoy_text = f"YOY change: {prev} → {latest} (Δ {change}, {pct})"

#     bullets = news_cat['summary'].apply(lambda x: f"- {x.strip()}").str.cat(sep="\n")
#     if not bullets.strip():
#         bullets = "No recent news articles available for this category."

#     prompt = f"""
#     Category: {category}
#     ### Claim Summary
#     - Total Claims: {claim_count}
#     - Avg Reported Loss: {avg_loss_text}
#     - Top Profession: {top_prof}
#     - YOY Change: {yoy_text}
#     ### News Feed Summary
#     {bullets}
#     ### Future Insights
#     Based on the above claims and news data, generate:
#     - 3–5 insights on current trends
#     - Future outlook for the next 6–12 months
#     - 3 actionable recommendations for risk owners
#     """
# ``    return _hf_summarize(prompt, max_length=300)



def generate_combined_category_summary(category: str, news_df: pd.DataFrame) -> str:
    """
    Generate a summary for a given news category using the _hf_summarize function.
    This version excludes any claims-related data and focuses solely on news content.
    """
    news_cat = news_df[news_df['keyword'] == category]
    bullets = news_cat['summary'].apply(lambda x: f"- {x.strip()}").str.cat(sep="\n")

    bullets = bullets[:1000]  # Truncate to first 1000 characters to avoid token limits

    if not bullets.strip():
        bullets = "No recent news articles available for this category."

    prompt = f"""
    As an insurance analyst, review the following news summaries related to the category: {category}.

    ### News Highlights
    {bullets}

    ### Analyst Summary
    Provide a concise (50–100 words) analysis covering:
    - Key incidents and their impact
    - Estimated or reported damages
    - Emerging trends and future risk outlook
    """

    return _hf_summarize(prompt, max_length=100)
