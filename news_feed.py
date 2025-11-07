import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

import pandas as pd
import pycountry
import plotly.express as px
import requests
import streamlit as st
from eventregistry import (
    EventRegistry,
    QueryArticlesIter,
    ReturnInfo,
    ArticleInfoFlags,
    SourceInfoFlags,
)

# =========================
# 0) CONFIG: API KEYS & RUNTIME
# =========================
# --- Recommended: use Streamlit secrets (add to .streamlit/secrets.toml)
# [secrets]
# EVENTREGISTRY_API_KEY = "hf_xxx"
# HUGGINGFACE_API_KEY   = "hf_xxx"
EVENTREGISTRY_API_KEY = st.secrets.get("EVENTREGISTRY_API_KEY") 
HUGGINGFACE_API_KEY   = st.secrets.get("HUGGINGFACE_API_KEY")   

# Optional corporate proxies (fill if needed; else leave empty)
PROXIES: Dict[str, str] = {
    # "http":  "http://user:pass@proxy-host:port",
    # "https": "http://user:pass@proxy-host:port",
}
REQUEST_KW = {"proxies": PROXIES, "timeout": 60}  # add verify="<path-to-ca>.pem" if SSL interception is used

# =========================
# 1) DEFAULT PARAMETERS (can be overridden via function call)
# =========================
# DEFAULT_KEYWORDS = ['Operational Error', 'Malicious Activity', 'Natural Disaster', 'Product/Service Failure' , 'Injury/Illness']
DEFAULT_KEYWORDS = ['Cyber', 'Fire', 'Flood', 'Malicious Activity' , 'Injury/Illness']
DEFAULT_MAX_PER_KEYWORD = 5
DEFAULT_START_DATE = "2025-10-08"
DEFAULT_END_DATE   = "2025-11-07"

# =========================
# 2) Country normalization / detection
# =========================
COUNTRY_SYNONYMS = {
    "U.S.": "United States",
    "US": "United States",
    "USA": "United States",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "Britain": "United Kingdom",
    "Great Britain": "United Kingdom",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Czech Republic": "Czechia",
    "Russia": "Russian Federation",
    "Iran": "Iran, Islamic Republic of",
    "Syria": "Syrian Arab Republic",
    "Vatican": "Holy See (Vatican City State)",
}

def _normalize_country(name: str) -> str:
    name = (name or "").strip()
    name = COUNTRY_SYNONYMS.get(name, name)
    try:
        country = pycountry.countries.lookup(name)
        return country.name
    except LookupError:
        return name


def _to_iso3(name: str) -> Optional[str]:
    if not name:
        return None
    try:
        c = pycountry.countries.lookup(name)
        return c.alpha_3
    except Exception:
        return None

_COUNTRIES = list(pycountry.countries)
_COUNTRY_NAMES = [c.name for c in _COUNTRIES]
for c in _COUNTRIES:
    if hasattr(c, "official_name"):
        _COUNTRY_NAMES.append(c.official_name)
_COUNTRY_NAMES += list(COUNTRY_SYNONYMS.keys()) + list(COUNTRY_SYNONYMS.values())
_COUNTRY_NAMES = list(dict.fromkeys([n for n in _COUNTRY_NAMES if n]))
_COUNTRY_PATTERNS = [(name, re.compile(rf"\b{re.escape(name)}\b", flags=re.IGNORECASE)) for name in _COUNTRY_NAMES]

def _detect_countries(text: str) -> List[str]:
    if not text:
        return []
    found = set()
    for name, pat in _COUNTRY_PATTERNS:
        if pat.search(text):
            found.add(_normalize_country(name))
    return sorted({_normalize_country(n) for n in found if n})

# =========================
# 3) Hugging Face summarization (robust)
# =========================
def _hf_summarize(text: str, model_name="facebook/bart-large-cnn", max_length=130, min_length=30) -> str:
    """
    Robust HF call with small retry logic.
    Default to a stable summarizer; include fallbacks; chunk long text.
    """
    if not text:
        return ""
    if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY.startswith("REPLACE_"):
        raise RuntimeError("HUGGINGFACE_API_KEY not set")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    models_to_try = [model_name, "sshleifer/distilbart-cnn-12-6", "philschmid/bart-large-cnn-samsum"]

    def chunk_text(t: str, limit: int = 2200) -> List[str]:
        if len(t) <= 2500:
            return [t]
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', t) if s.strip()]
        parts, cur = [], ""
        for s in sents:
            if len(cur) + len(s) + 1 <= limit:
                cur += (s + " ")
            else:
                parts.append(cur.strip()); cur = s + " "
        if cur:
            parts.append(cur.strip())
        return parts

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
                except requests.HTTPError as http_e:
                    last_exc = http_e; time.sleep(0.3); continue
                except Exception as e:
                    last_exc = e; time.sleep(0.3); continue

                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        txt = first.get("summary_text") or first.get("generated_text") or ""
                        summaries.append(txt)
                    else:
                        summaries.append(" ".join([str(x) for x in data]))
                elif isinstance(data, dict) and data.get("error"):
                    last_exc = RuntimeError(data.get("error")); continue
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

    return " ".join([s for s in summaries if s]).strip()

def _summarize_text(text, max_length=130):
    if not text:
        return ""
    try:
        return _hf_summarize(text, model_name="facebook/bart-large-cnn", max_length=max_length)
    except Exception:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return " ".join(sents[:2])

# =========================
# 4) EventRegistry fetch (iterator + ReturnInfo)
# =========================
def _clean_text(*parts):
    text = " ".join([p for p in parts if p])
    return re.sub(r"\s+", " ", text).strip()

def _parse_date_str(d: str) -> datetime:
    try:
        return datetime.fromisoformat((d or "").replace("Z", "+00:00"))
    except Exception:
        return datetime(1970,1,1, tzinfo=timezone.utc)

@st.cache_data(show_spinner=False, ttl=30*60)
def _fetch_news_df(_keywords: tuple, _start: str, _end: str, _max_per: int) -> pd.DataFrame:
    if not EVENTREGISTRY_API_KEY or EVENTREGISTRY_API_KEY.startswith("REPLACE_"):
        raise RuntimeError("EVENTREGISTRY_API_KEY not set")
    er = EventRegistry(apiKey=EVENTREGISTRY_API_KEY, allowUseOfArchive=False)
    ret_info = ReturnInfo(
        articleInfo=ArticleInfoFlags(bodyLen=-1, concepts=False, categories=False, authors=True),
        sourceInfo=SourceInfoFlags()
    )

    rows: List[dict] = []
    for kw in _keywords:
        q = QueryArticlesIter(
            keywords=kw,
            dateStart=_start,
            dateEnd=_end,
            isDuplicateFilter="skipDuplicates",
            lang="eng"
        )
        for art in q.execQuery(er, returnInfo=ret_info, sortBy="date", sortByAsc=False, maxItems=_max_per):
            try:
                title = art.get("title") or ""
                url = art.get("url") or ""
                source = (art.get("source") or {}).get("title") or (art.get("source") or {}).get("uri") or ""
                date = art.get("dateTime", "") or art.get("date", "")
                body = art.get("body") or ""
                authors = ", ".join((art.get("authors") or [])) if isinstance(art.get("authors"), list) else (art.get("authors") or "")

                text_for_summary = _clean_text(title, body)
                countries = _detect_countries(text_for_summary)
                summary = _summarize_text(text_for_summary, max_length=130)

                rows.append({
                    "date": date, "keyword": kw, "title": title, "url": url, "source": source,
                    "authors": authors, "countries": countries, "summary": summary, "body": body
                })
            except Exception:
                continue

    # de-duplicate by (url, title)
    seen, out = set(), []
    for r in rows:
        k = ((r.get("url") or "").strip().lower(), (r.get("title") or "").strip().lower())
        if k in seen: continue
        seen.add(k); out.append(r)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["date_parsed"] = df["date"].apply(_parse_date_str)
    df["countries_csv"] = df["countries"].apply(lambda x: ", ".join(x) if isinstance(x, list) else (x or ""))
    return df.sort_values("date_parsed", ascending=False).reset_index(drop=True)

# =========================
# 5) Roll-up + viz helpers
# =========================
def _build_bullets(df: pd.DataFrame, max_chars: int = 6000) -> str:
    rows, total = [], 0
    for _, r in df.iterrows():
        title = (r.get("title") or "").strip()
        summ = (r.get("summary") or "").strip()
        cn_list = (r.get("countries") or [])[:2]
        cn = ", ".join(cn_list) if cn_list else ""
        piece = f"- {title}"
        if summ: piece += f" — {summ}"
        if cn:   piece += f" (Countries: {cn})"
        if total + len(piece) + 1 > max_chars: break
        rows.append(piece); total += len(piece) + 1
    return "\n".join(rows)

def _local_rollup_summary(df_cat: pd.DataFrame, category_name: str, top_n: int = 3) -> str:
    if df_cat.empty:
        return "No articles available."
    bullets = []
    n, uniq_src = len(df_cat), df_cat["source"].nunique()
    span = f"{df_cat['date_parsed'].min().date()} → {df_cat['date_parsed'].max().date()}"
    bullets.append(f"- **Coverage:** {n} articles from {uniq_src} sources ({span}).")
    try:
        cc = (df_cat.explode("countries").assign(country_norm=lambda d: d["countries"].apply(_normalize_country)))
        cc = cc[cc["country_norm"].notna() & (cc["country_norm"] != "")]
        cc = cc.groupby("country_norm").size().sort_values(ascending=False).head(top_n)
        if not cc.empty:
            bullets.append("- **Top geographies:** " + ", ".join([f"{k} ({int(v)})" for k, v in cc.items()]))
    except Exception:
        pass
    src = df_cat["source"].value_counts().head(top_n)
    if not src.empty:
        bullets.append("- **Top sources:** " + ", ".join([f"{k} ({v})" for k, v in src.items()]))
    heads = [t for t in (df_cat["title"].dropna().tolist()[:top_n]) if t.strip()]
    if heads:
        bullets.append("- **Notable headlines:** " + "; ".join(heads))
    bullets.append("_(Offline roll‑up. Hugging Face call failed.)_")
    return "\n".join(bullets)

@st.cache_data(show_spinner=False, ttl=60*60)
def _llm_category_summary(category: str, bullets_text: str, date_label: str, df_cat: pd.DataFrame) -> str:
    if not bullets_text.strip():
        return "No articles available to summarize."
    text = bullets_text[:6000]
    prompt = (
        f"Category: {category}\n"
        f"Date window: {date_label}\n\n"
        f"Below are bullets (title — per-article summary — countries). "
        f"Produce 3–5 crisp bullets and list the top 3 geographies by mentions. Be factual.\n\n"
        f"{text}"
    )
    try:
        return _hf_summarize(prompt, model_name="facebook/bart-large-cnn", max_length=220, min_length=80)
    except Exception:
        return _local_rollup_summary(df_cat, category_name=category, top_n=3)

def _country_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["country", "iso3", "count"])
    ex = df.explode("countries")
    ex = ex[ex["countries"].notna() & (ex["countries"] != "")]
    if ex.empty:
        return pd.DataFrame(columns=["country", "iso3", "count"])
    ex["country_norm"] = ex["countries"].apply(_normalize_country)
    ex["iso3"] = ex["country_norm"].apply(_to_iso3)
    grouped = (ex.groupby(["country_norm", "iso3"], dropna=True)
               .size().reset_index(name="count").rename(columns={"country_norm": "country"}))
    return grouped[grouped["iso3"].notna()].sort_values("count", ascending=False)

def _choropleth(counts_df: pd.DataFrame, title: str):
    if counts_df.empty:
        st.info("No country mentions found for this selection.")
        return
    fig = px.choropleth(
        counts_df, locations="iso3", color="count", hover_name="country",
        color_continuous_scale="Blues", projection="natural earth", title=title
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

def _linkify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def mk(row):
        t = row.get("title") or "link"
        u = row.get("url") or ""
        return f"<a href='{u}' target='_blank'>{t}</a>" if u else t
    df["Article"] = df.apply(mk, axis=1)
    return df

# =========================
# 6) PUBLIC RENDER FUNCTION
# =========================
def render_news_feed_insights(
    keywords: List[str] = None,
    start_date: str = None,
    end_date: str = None,
    max_per_keyword: int = None,
    section_title: str = "News Feed Insights",
    tab_titles: List[str] = None,
):
    """
    Render the full News Feed Insights UI into the current container (e.g., inside your tab).
    Customize via parameters or rely on defaults.
    """
    keywords = keywords or DEFAULT_KEYWORDS
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or DEFAULT_END_DATE
    max_per_keyword = max_per_keyword or DEFAULT_MAX_PER_KEYWORD
    tab_titles = tab_titles or ["Overview", "LLM Summary"]

    # Guardrails for keys
    if not EVENTREGISTRY_API_KEY or EVENTREGISTRY_API_KEY.startswith("REPLACE_"):
        st.error("EVENTREGISTRY_API_KEY is not set. Use st.secrets or edit news_feed.py.")
        return
    if not HUGGINGFACE_API_KEY or HUGGINGFACE_API_KEY.startswith("REPLACE_"):
        st.warning("HUGGINGFACE_API_KEY is not set. LLM roll-up will use offline fallback.")

    # 1) Data (cached in session under a unique key fingerprint)
    cache_key = f"news_df_cache::{hash((tuple(keywords), start_date, end_date, int(max_per_keyword)))}"
    if cache_key not in st.session_state:
        with st.spinner("Fetching news & generating summaries..."):
            st.session_state[cache_key] = _fetch_news_df(tuple(keywords), start_date, end_date, int(max_per_keyword))
    df = st.session_state.get(cache_key, pd.DataFrame())

    # 2) Header & KPIs
    st.subheader(section_title)
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles", f"{len(df):,}")
    col2.metric("Sources", df["source"].nunique() if not df.empty else 0)
    if not df.empty:
        col3.metric("Date Span", f"{df['date_parsed'].min().date()} → {df['date_parsed'].max().date()}")
    else:
        col3.metric("Date Span", "—")

    st.divider()

    # 3) Sidebar filters (local to this component)
    
    st.markdown("### News Feed Filters")
    if df.empty:
        st.caption("No data to filter.")
        category = "All"; date_enabled = False; d1 = d2 = None
    else:
        categories = sorted([c for c in df["keyword"].dropna().unique() if str(c).strip()])
        category = st.selectbox("Category", ["All"] + categories, index=0, key=f"nf_cat::{cache_key}")
        min_dt, max_dt = df["date_parsed"].min(), df["date_parsed"].max()
        if pd.isna(min_dt) or pd.isna(max_dt):
            date_enabled = False; st.caption("No valid dates detected."); d1 = d2 = None
        else:
            date_enabled = True
            d1, d2 = st.date_input(
                "Date range",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(), max_value=max_dt.date(),
                key=f"nf_dates::{cache_key}"
            )

    # 4) Apply filters once, share across inner tabs
    df_sel = df.copy()
    if category != "All":
        df_sel = df_sel[df_sel["keyword"] == category]
    if date_enabled and not df_sel.empty:
        start_ts = pd.Timestamp(d1).tz_localize("UTC")
        end_ts   = pd.Timestamp(d2).tz_localize("UTC") + pd.Timedelta(days=1)
        mask = (df_sel["date_parsed"] >= start_ts) & (df_sel["date_parsed"] < end_ts)
        df_sel = df_sel[mask]

    # 5) Inner tabs
    t_overview, t_summary = st.tabs(tab_titles)

    with t_overview:
        st.subheader("🗺️ Articles by Country")
        counts = _country_counts(df_sel)
        _choropleth(counts, title=f"{'All' if category=='All' else category}: mentions by country")

        st.divider()
        st.subheader("📄 Articles")
        if df_sel.empty:
            st.info("No rows to show.")
        else:
            table = _linkify(df_sel)[["Article", "date", "keyword", "source", "countries_csv", "summary"]]
            table = table.rename(columns={"countries_csv": "countries"})
            st.dataframe(table, use_container_width=True, hide_index=True)

    with t_summary:
        st.subheader("🧠 Category Summary")
        if df_sel.empty:
            st.info("No articles for this selection.")
        else:
            bullets = _build_bullets(df_sel, max_chars=6000)
            date_lbl = f"from {df_sel['date_parsed'].min().date()} to {df_sel['date_parsed'].max().date()}"
            with st.spinner("Generating AI roll-up..."):
                summary = _llm_category_summary(
                    "All categories" if category == "All" else category,
                    bullets, date_lbl, df_sel
                )
            st.markdown(summary)


# news_feed.py
# import re
# from datetime import datetime, timedelta
# from typing import Optional, List, Dict
# import pandas as pd
# import pycountry
# import plotly.express as px
# import requests
# import streamlit as st
# from eventregistry import EventRegistry, QueryArticlesIter, ReturnInfo, ArticleInfoFlags, SourceInfoFlags

# from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity



# # =========================
# # CONFIG: API KEYS
# # =========================
# EVENTREGISTRY_API_KEY = st.secrets["EVENTREGISTRY_API_KEY"]
# HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

# # =========================
# # Claims Extraction
# # =========================

# def extract_claims_info(text: str) -> Dict[str, Optional[str]]:
#     claims_data = {
#         "claim_amount": None,
#         "loss_ratio": None,
#         "volume": None,
#         "severity": None,
#         "profession": None,
#         "loss_amount": None  # NEW FIELD
#     }

#     # Existing claim amount extraction
#     match = re.search(r"claim amount of \$([\d\.]+)\s*(billion|million)?", text, re.IGNORECASE)
#     if match:
#         amount = float(match.group(1))
#         unit = match.group(2)
#         if unit == "billion":
#             amount *= 1_000_000_000
#         elif unit == "million":
#             amount *= 1_000_000
#         claims_data["claim_amount"] = amount

#     # NEW: Loss amount extraction for catastrophic risks
#     match = re.search(r"(reported|estimated) losses? of \$([\d\.]+)\s*(billion|million)?", text, re.IGNORECASE)
#     if match:
#         amount = float(match.group(2))
#         unit = match.group(3)
#         if unit == "billion":
#             amount *= 1_000_000_000
#         elif unit == "million":
#             amount *= 1_000_000
#         claims_data["loss_amount"] = amount

#     # Existing logic for other fields...
#     match = re.search(r"loss ratio.*?(\d+)%", text)
#     if match:
#         claims_data["loss_ratio"] = int(match.group(1))

#     match = re.search(r"(\d{1,3}(,\d{3})*)\s+claims", text)
#     if match:
#         claims_data["volume"] = int(match.group(1).replace(",", ""))

#     if "severity" in text.lower():
#         claims_data["severity"] = "High"

#     match = re.search(r"especially in ([\w\s]+) professions", text)
#     if match:
#         claims_data["profession"] = match.group(1).strip()

#     return claims_data


# # =========================
# # Summarization with Claims Info
# # =========================
# def format_for_summary(article: str, claims_data: Dict[str, Optional[str]]) -> str:
#     return f"""
# Article:
# {article}

# Extracted Claims Info:
# - Claim Amount: {claims_data['claim_amount']}
# - Loss Ratio: {claims_data['loss_ratio']}%
# - Volume: {claims_data['volume']} claims
# - Severity: {claims_data['severity']}
# - Most Active Profession: {claims_data['profession']}

# Summarize the article with emphasis on the claims-level insights.
# """

# def hf_summarize(text: str) -> str:
#     headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
#     payload = {
#         "inputs": text,
#         "parameters": {"max_length": 220, "min_length": 80}
#     }
#     url = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-cnn"
#     try:
#         response = requests.post(url, headers=headers, json=payload, timeout=60)
#         response.raise_for_status()
#         data = response.json()
#         return data[0].get("summary_text", "")
#     except Exception:
#         return "Summary generation failed."

# # =========================
# # Country Detection
# # =========================
# def detect_countries(text: str) -> List[str]:
#     countries = []
#     for country in pycountry.countries:
#         if re.search(rf"\b{re.escape(country.name)}\b", text, re.IGNORECASE):
#             countries.append(country.name)
#     return countries


# # =========================
# # Deduplication Function
# # ========================


# model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and fast

# def deduplicate_articles_semantic(df, threshold=0.85):
#     summaries = df['summary'].tolist()
#     embeddings = model.encode(summaries)
#     sim_matrix = cosine_similarity(embeddings)

#     keep_indices = []
#     seen = set()
#     for i in range(len(summaries)):
#         if i in seen:
#             continue
#         keep_indices.append(i)
#         for j in range(i+1, len(summaries)):
#             if sim_matrix[i][j] > threshold:
#                 seen.add(j)
#     return df.iloc[keep_indices].reset_index(drop=True)



# # =========================
# # Visualization
# # =========================
# def plot_claims_map(df: pd.DataFrame):
#     exploded = df.explode("countries")
    
#     exploded["loss_amount"] = exploded["claims"].apply(lambda x: x.get("loss_amount", 0))
#     grouped = exploded.groupby("countries")["loss_amount"].sum().reset_index()

#     grouped["iso3"] = grouped["countries"].apply(lambda name: pycountry.countries.get(name=name).alpha_3 if pycountry.countries.get(name=name) else None)
#     grouped = grouped.dropna(subset=["iso3"])
#     fig = px.choropleth(
#         grouped,
#         locations="iso3",
#         color="loss_amount",
#         hover_name="countries",
#         color_continuous_scale="Reds",
#         title="Reported Loss Amount by Country (Catastrophic Risks)"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     # KPI Cards
    
#     total_loss_amount = exploded["loss_amount"].sum()
#     # col1.metric("Total Reported Loss Amount", f"${total_loss_amount:,.0f}")

#     average_loss_ratio = exploded["claims"].apply(lambda x: x.get("loss_ratio", 0)).mean()
#     st.subheader("📊 Key Metrics")
#     col1, col2 = st.columns(2)
#     col1.metric("Total Reported Claim Amount", f"${total_loss_amount:,.0f}")
#     col2.metric("Average Loss Ratio", f"{average_loss_ratio:.2f}%")

# # =========================
# # Streamlit UI
# # =========================
# def render_news_feed_insights():
#     st.title("📰 Claims-Level News Feed Insights")
    
#     categories = [
#         "Operational Error",
#         "Malicious Activity",
#         "Natural Disaster",
#         "Product/Service Failure",
#         "Injury/Illness"
#     ]
#     selected_categories = st.multiselect("Select Incident Categories", categories, default=categories)

#     end_date = datetime.today()
#     start_date = end_date - timedelta(days=30)
#     start_str = start_date.strftime('%Y-%m-%d')
#     end_str = end_date.strftime('%Y-%m-%d')
#     max_items = st.slider("Max articles per keyword", 1, 10, 5)

#     if st.button("Fetch & Analyze"):
#         with st.spinner("Fetching articles and generating summaries..."):
#             er = EventRegistry(apiKey=EVENTREGISTRY_API_KEY)
#             ret_info = ReturnInfo(
#                 articleInfo=ArticleInfoFlags(bodyLen=-1, concepts=False, categories=False, authors=True),
#                 sourceInfo=SourceInfoFlags()
#             )
#             rows = []
#             for category in selected_categories:
#                 q = QueryArticlesIter(
#                     keywords=category,
#                     dateStart=start_str,
#                     dateEnd=end_str,
#                     isDuplicateFilter="skipDuplicates",
#                     lang="eng"
#                 )
#                 for art in q.execQuery(er, returnInfo=ret_info, sortBy="date", sortByAsc=False, maxItems=max_items):
#                     title = art.get("title", "")
#                     body = art.get("body", "")
#                     claims = extract_claims_info(body)
#                     countries = detect_countries(body)
#                     summary_prompt = format_for_summary(body, claims)
#                     summary = hf_summarize(summary_prompt)
#                     rows.append({
#                         "title": title,
#                         "date": art.get("dateTime", ""),
#                         "source": art.get("source", {}).get("title", ""),
#                         "url": art.get("url", ""),
#                         "summary": summary,
#                         "countries": countries,
#                         "claims": claims
#                     })
#             df = pd.DataFrame(rows)
            
#             # Remove rows where summary generation failed
#             df = df[df['summary'] != "Summary generation failed."]

            
#             # Apply deduplication
#             if not df.empty:
#                 df = deduplicate_articles_semantic(df)

#             df = df.drop_duplicates(subset=['countries'])

#             st.success(f"Fetched {len(df)} articles.")

#             st.subheader("Summaries")
#             for _, row in df.iterrows():
#                 st.markdown(f"[{row['title']}]({row['url']})")
#                 st.text_area("Summary", row['summary'], height=150, key=row['url'])

#             # st.subheader("Article Table")
#             # df_display = df.copy()
#             # df_display["title"] = df_display.apply(lambda r: f"[{r['title']}]({r['url']})", axis=1)
#             # st.write(df_display[["title", "date", "source"]].to_html(escape=False, index=False), unsafe_allow_html=True)

#             plot_claims_map(df)


