import os
import sqlite3
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from io import BytesIO
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import yfinance as yf
from pytrends.request import TrendReq

# ------------------------- CONFIGURATION -------------------------
st.set_page_config(page_title="Profit Pulse Intelligence Hub", layout="wide", page_icon="💰")

LOG_FILE = "app_activity.log"
DB_PATH = "growth_engine.db"
DEFAULT_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
DEFAULT_CRYPTO = ["bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin"]
DEFAULT_KEYWORDS = ["ai tools", "side hustle", "drop shipping", "print on demand", "keto recipes", "stock alerts"]
PREMIUM_RESULT_LIMIT = 30
FREE_RESULT_LIMIT = 6
ALERT_THRESHOLD_PERCENT = 4
PRICE_CHECK_API = "https://dummyjson.com/products/search"
COINGECKO_API_BASE = "https://api.coingecko.com/api/v3"
GOOGLE_TREND_TIMEFRAME = "now 7-d"
STRIPE_WEBHOOK_URL = os.getenv("STRIPE_WEBHOOK_URL", "https://hooks.stripe.com/test")
AMAZON_AFFILIATE_URL = os.getenv("AMAZON_AFFILIATE_URL", "https://www.amazon.com?tag=affiliate-id")
BINANCE_AFFILIATE_URL = os.getenv("BINANCE_AFFILIATE_URL", "https://accounts.binance.com/en/register?ref=AFFILIATE")
COURSE_AFFILIATE_URL = os.getenv("COURSE_AFFILIATE_URL", "https://www.udemy.com/course/startup/?ref=AFFILIATE")
DEFAULT_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://profitpulse.app")

# ------------------------- LOGGING SETUP -------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ------------------------- DATABASE UTILITIES -------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                email TEXT,
                referral_code TEXT,
                referred_by TEXT,
                is_premium INTEGER DEFAULT 0,
                created_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT,
                details TEXT,
                created_at TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                referral_code TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    except sqlite3.Error as err:
        logging.error(f"Database initialization failed: {err}")
    finally:
        conn.close()


def upsert_user(username: str, email: str, referral_code: Optional[str], referred_by: Optional[str], is_premium: bool) -> None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, email, referral_code, referred_by, is_premium, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                email,
                referral_code,
                referred_by,
                int(is_premium),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    except sqlite3.Error as err:
        logging.error(f"Failed to upsert user: {err}")
    finally:
        conn.close()


def update_user_premium(username: str, is_premium: bool) -> None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET is_premium = ? WHERE username = ?",
            (int(is_premium), username),
        )
        conn.commit()
    except sqlite3.Error as err:
        logging.error(f"Failed to update premium status: {err}")
    finally:
        conn.close()


def log_activity(username: str, action: str, details: str) -> None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO activity_logs (username, action, details, created_at) VALUES (?, ?, ?, ?)",
            (username, action, details, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.Error as err:
        logging.error(f"Failed to log activity: {err}")
    finally:
        conn.close()


def save_lead(email: str, referral_code: Optional[str]) -> None:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO leads (email, referral_code, created_at) VALUES (?, ?, ?)",
            (email, referral_code, datetime.utcnow().isoformat()),
        )
        conn.commit()
    except sqlite3.Error as err:
        logging.error(f"Failed to save lead: {err}")
    finally:
        conn.close()


# ------------------------- SESSION STATE -------------------------
def initialize_state() -> None:
    if "username" not in st.session_state:
        st.session_state.username = "Guest"
    if "email" not in st.session_state:
        st.session_state.email = ""
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "is_premium" not in st.session_state:
        st.session_state.is_premium = False
    if "referral_code" not in st.session_state:
        st.session_state.referral_code = None
    if "unlock_attempts" not in st.session_state:
        st.session_state.unlock_attempts = 0
    if "generated_referral" not in st.session_state:
        st.session_state.generated_referral = None


# ------------------------- DATA FETCHING -------------------------
@st.cache_data(show_spinner=False, ttl=600)
def fetch_stock_data(symbols: List[str], period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            threads=False,
        )
        frames = []
        if isinstance(data.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    sym_df = data[symbol].reset_index()
                    sym_df["Symbol"] = symbol
                    frames.append(sym_df)
                except KeyError:
                    logging.warning(f"Symbol {symbol} missing in stock dataset")
        else:
            data = data.reset_index()
            data["Symbol"] = ",".join(symbols)
            frames.append(data)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.rename(columns={"index": "Date"})
            return combined
    except Exception as err:
        logging.error(f"Stock data fetch failed: {err}")
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def fetch_crypto_prices(coin_ids: List[str]) -> pd.DataFrame:
    if not coin_ids:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"{COINGECKO_API_BASE}/simple/price",
            params={
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = []
        for coin in coin_ids:
            data = payload.get(coin)
            if data:
                rows.append(
                    {
                        "Coin": coin.title(),
                        "Price": data.get("usd"),
                        "Change24h": data.get("usd_24h_change"),
                    }
                )
        return pd.DataFrame(rows)
    except Exception as err:
        logging.error(f"Crypto price fetch failed: {err}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def fetch_trending_crypto() -> pd.DataFrame:
    try:
        resp = requests.get(f"{COINGECKO_API_BASE}/search/trending", timeout=10)
        resp.raise_for_status()
        payload = resp.json().get("coins", [])
        records = []
        for item in payload:
            coin = item.get("item", {})
            records.append(
                {
                    "Coin": coin.get("name"),
                    "Symbol": coin.get("symbol"),
                    "Market Rank": coin.get("market_cap_rank"),
                    "Score": coin.get("score"),
                    "Price (BTC)": coin.get("price_btc"),
                }
            )
        return pd.DataFrame(records)
    except Exception as err:
        logging.error(f"Trending crypto fetch failed: {err}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_google_trends(keywords: List[str]) -> pd.DataFrame:
    if not keywords:
        return pd.DataFrame()
    try:
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(keywords, timeframe=GOOGLE_TREND_TIMEFRAME)
        interest_over_time = pytrends.interest_over_time()
        if interest_over_time.empty:
            return pd.DataFrame()
        interest_over_time = interest_over_time.reset_index()
        melted = interest_over_time.melt(id_vars=["date"], var_name="Keyword", value_name="Interest")
        melted = melted[melted["Keyword"] != "isPartial"]
        return melted
    except Exception as err:
        logging.error(f"Google Trends fetch failed: {err}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=900)
def fetch_product_prices(keyword: str) -> pd.DataFrame:
    if not keyword:
        return pd.DataFrame()
    try:
        resp = requests.get(PRICE_CHECK_API, params={"q": keyword}, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("products", [])
        rows = []
        for item in items:
            rows.append(
                {
                    "Product": item.get("title"),
                    "Price": item.get("price"),
                    "Discount": item.get("discountPercentage"),
                    "Rating": item.get("rating"),
                    "Image URL": item.get("thumbnail"),
                }
            )
        return pd.DataFrame(rows)
    except Exception as err:
        logging.error(f"Product price fetch failed: {err}")
        return pd.DataFrame()


# ------------------------- BUSINESS LOGIC -------------------------
def calculate_stock_signals(stock_df: pd.DataFrame) -> pd.DataFrame:
    if stock_df.empty:
        return pd.DataFrame()
    signals = []
    for symbol, data in stock_df.groupby("Symbol"):
        data = data.sort_values("Date")
        if len(data) < 2:
            continue
        latest_close = data.iloc[-1]["Close"]
        prev_close = data.iloc[-2]["Close"]
        if prev_close == 0 or pd.isna(prev_close):
            continue
        percent_change = ((latest_close - prev_close) / prev_close) * 100
        ma5 = data["Close"].tail(5).mean()
        ma20 = data["Close"].tail(20).mean()
        momentum = "Bullish" if ma5 > ma20 else "Bearish"
        alert = "BUY" if percent_change > ALERT_THRESHOLD_PERCENT and momentum == "Bullish" else "WATCH"
        if percent_change < -ALERT_THRESHOLD_PERCENT:
            alert = "SELL"
        signals.append(
            {
                "Asset": symbol,
                "Latest Close": round(float(latest_close), 2),
                "Daily %": round(float(percent_change), 2),
                "MA5": round(float(ma5), 2) if not np.isnan(ma5) else None,
                "MA20": round(float(ma20), 2) if not np.isnan(ma20) else None,
                "Momentum": momentum,
                "Action": alert,
            }
        )
    return pd.DataFrame(signals)


def calculate_crypto_signals(crypto_df: pd.DataFrame, trending_df: pd.DataFrame) -> pd.DataFrame:
    if crypto_df.empty:
        return pd.DataFrame()
    merged = crypto_df.copy()
    merged["Signal"] = merged["Change24h"].apply(
        lambda x: "BUY" if x and x > ALERT_THRESHOLD_PERCENT else ("SELL" if x and x < -ALERT_THRESHOLD_PERCENT else "WATCH")
    )
    if not trending_df.empty:
        trending_symbols = [s.lower() for s in trending_df["Symbol"].dropna().astype(str).tolist()]
        merged["Trending"] = merged["Coin"].str.lower().apply(
            lambda c: "🔥" if any(c.startswith(sym.lower()) for sym in trending_symbols) else ""
        )
    else:
        merged["Trending"] = ""
    merged["Change24h"] = merged["Change24h"].astype(float).round(2)
    merged["Price"] = merged["Price"].astype(float).round(4)
    return merged


def calculate_keyword_opportunities(trend_df: pd.DataFrame) -> pd.DataFrame:
    if trend_df.empty:
        return pd.DataFrame()
    opportunities = []
    for keyword, group in trend_df.groupby("Keyword"):
        recent = group.tail(7)
        if recent.empty:
            continue
        start = recent.iloc[0]["Interest"]
        end = recent.iloc[-1]["Interest"]
        if pd.isna(start) or pd.isna(end):
            continue
        change = end - start
        pct = (change / start * 100) if start else 0
        if pct >= 15:
            urgency = "Exploding"
        elif pct >= 5:
            urgency = "Rising"
        else:
            urgency = "Stable"
        opportunities.append(
            {
                "Keyword": keyword,
                "7d Change": round(float(change), 2),
                "7d %": round(float(pct), 2),
                "Status": urgency,
                "Suggested Offer": f"Launch content + affiliate funnel for '{keyword}'",
            }
        )
    df = pd.DataFrame(opportunities)
    if not df.empty:
        df = df.sort_values("7d %", ascending=False)
    return df


def limit_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    limit = PREMIUM_RESULT_LIMIT if st.session_state.is_premium else FREE_RESULT_LIMIT
    return df.head(limit)


def generate_referral_code(username: str) -> str:
    base = f"{username}-{uuid.uuid4().hex[:6]}"
    return base.upper()


def create_referral_link(referral_code: str) -> str:
    params = st.experimental_get_query_params()
    params = {key: values[0] if isinstance(values, list) and values else values for key, values in params.items()}
    params["ref"] = referral_code
    query = urlencode(params)
    return f"{DEFAULT_BASE_URL}?{query}" if query else f"{DEFAULT_BASE_URL}?ref={referral_code}"


def prepare_export_payload(sections: Dict[str, pd.DataFrame]) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sections.items():
            if df.empty:
                continue
            safe_name = name[:31] if name else "Sheet"
            df.to_excel(writer, sheet_name=safe_name, index=False)
    output.seek(0)
    return output


# ------------------------- USER INTERFACE -------------------------
def sidebar_ui():
    st.sidebar.header("Growth Engine Access")
    with st.sidebar.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", value=st.session_state.username or "")
        email = st.text_input("Email", value=st.session_state.email or "", help="Used for delivering premium reports and deal alerts.")
        referral_code = st.text_input("Referral Code", value=st.session_state.referral_code or "")
        submitted = st.form_submit_button("Log In / Register")
        if submitted:
            st.session_state.username = username or "Guest"
            st.session_state.email = email
            st.session_state.is_authenticated = bool(username and email)
            st.session_state.referral_code = referral_code or None
            if st.session_state.is_authenticated:
                user_referral = generate_referral_code(username)
                st.session_state.generated_referral = user_referral
                upsert_user(username, email, user_referral, referral_code or None, st.session_state.is_premium)
                log_activity(username, "login", "User logged in or registered")
                st.success("Access granted. Premium features available after upgrade.")
            else:
                st.warning("Enter both username and email to unlock full dashboard capabilities.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Marketing Automation")
    with st.sidebar.form("lead_capture"):
        lead_email = st.text_input("Join the Deal List", key="lead_email")
        lead_ref = st.text_input("Your Referral Code", key="lead_ref")
        capture = st.form_submit_button("Save Email")
        if capture and lead_email:
            save_lead(lead_email, lead_ref or None)
            st.success("Email captured! Expect monetized alerts soon.")
            log_activity(st.session_state.username, "lead_capture", lead_email)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Premium Unlock")
    if st.session_state.is_premium:
        st.sidebar.success("Premium tier active. Unlimited intelligence unlocked.")
    else:
        if st.sidebar.button("Unlock via Stripe Checkout", help="Simulated Stripe integration"):
            st.session_state.is_premium = True
            st.session_state.unlock_attempts += 1
            update_user_premium(st.session_state.username, True)
            log_activity(st.session_state.username, "upgrade", f"Stripe webhook: {STRIPE_WEBHOOK_URL}")
            st.sidebar.success("Premium activated! Refresh sections for expanded data.")
        st.sidebar.info("Premium users receive unlimited signals, white-label exports, and high-priority alerts.")


# ------------------------- MAIN CONTENT -------------------------
def main_content():
    st.title("💰 Profit Pulse Intelligence Hub")
    st.caption("AI-assisted trade, keyword, and product arbitrage radar engineered for recurring revenue.")

    st.markdown(
        """
        <div style="background-color:#101820;padding:14px;border-radius:10px;color:white;">
            <h3 style="margin-bottom:8px;">Monetize smarter:</h3>
            <ul>
                <li>Generate daily buy/sell signals aligned with affiliate brokerage links.</li>
                <li>Discover exploding keywords, then upsell SEO packages or info products.</li>
                <li>Track ecommerce discounts for instant arbitrage listings on marketplaces.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Logged User", st.session_state.username or "Guest")
    with col2:
        st.metric("Premium Status", "Active" if st.session_state.is_premium else "Free")
    with col3:
        st.metric("Unlock Attempts", st.session_state.unlock_attempts)

    st.subheader("Market Scanner")
    selected_stocks = st.multiselect("Stock Symbols", DEFAULT_STOCKS, default=DEFAULT_STOCKS[:4])
    selected_coins = st.multiselect("Crypto Assets", DEFAULT_CRYPTO, default=DEFAULT_CRYPTO[:4])
    selected_keywords = st.multiselect("Keyword Seeds", DEFAULT_KEYWORDS, default=DEFAULT_KEYWORDS[:3])
    search_keyword = st.text_input("Ecommerce Product Keyword", value="wireless earbuds", help="Identify arbitrage-ready deals.")
    timeframe = st.selectbox("Stock Timeframe", ["1mo", "3mo", "6mo"], index=0)
    refresh_clicked = st.button("Refresh Intelligence", type="primary")
    if refresh_clicked:
        log_activity(
            st.session_state.username,
            "refresh",
            f"stocks={selected_stocks}, crypto={selected_coins}, keywords={selected_keywords}, product={search_keyword}",
        )

    with st.spinner("Collecting financial and marketing intelligence..."):
        stock_df = fetch_stock_data(selected_stocks, period=timeframe)
        crypto_df = fetch_crypto_prices(selected_coins)
        trending_crypto_df = fetch_trending_crypto()
        trend_df = fetch_google_trends(selected_keywords)
        product_df = fetch_product_prices(search_keyword)

    st.markdown("### Trading Signals")
    stock_signals = calculate_stock_signals(stock_df)
    crypto_signals = calculate_crypto_signals(crypto_df, trending_crypto_df)
    st.write("**Equity Opportunities**")
    st.dataframe(limit_results(stock_signals), use_container_width=True)
    st.write("**Crypto Momentum**")
    st.dataframe(limit_results(crypto_signals), use_container_width=True)

    if not stock_df.empty:
        chart_df = stock_df[["Date", "Close", "Symbol"]]
        fig = px.line(chart_df, x="Date", y="Close", color="Symbol", title="Price Action Snapshot")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### SEO Growth Opportunities")
    keyword_opps = calculate_keyword_opportunities(trend_df)
    st.dataframe(limit_results(keyword_opps), use_container_width=True)

    if not trend_df.empty:
        interest_chart = px.line(trend_df, x="date", y="Interest", color="Keyword", title="Search Demand Trajectories")
        st.plotly_chart(interest_chart, use_container_width=True)

    st.markdown("### Ecommerce Discount Radar")
    st.dataframe(limit_results(product_df), use_container_width=True)

    col_export, col_report = st.columns(2)
    with col_export:
        export_data = {
            "Stock Signals": stock_signals,
            "Crypto Signals": crypto_signals,
            "Keyword Opportunities": keyword_opps,
            "Product Deals": product_df,
        }
        export_file = prepare_export_payload(export_data)
        st.download_button(
            "Download Intelligence Pack (Excel)",
            data=export_file,
            file_name="profit_pulse_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with col_report:
        st.info(
            "Premium members can automate delivery: connect Stripe via webhook."
            if st.session_state.is_premium
            else "Upgrade to export unlimited data & unlock API automations."
        )
        if not st.session_state.is_premium and st.button("Email Me The Full Report (Premium)"):
            st.warning("Premium plan required. Click Unlock in the sidebar to activate.")
            log_activity(st.session_state.username, "premium_prompt", "Requested full report")
        elif st.session_state.is_premium:
            if st.button("Send Report via Email"):
                st.success(f"Report queued for delivery to {st.session_state.email or 'your inbox'} (simulation).")
                log_activity(st.session_state.username, "report_email", st.session_state.email or "no-email")

    st.markdown("### Monetization Playbooks")
    monetization_cols = st.columns(3)
    monetization_cols[0].markdown(
        f"""
        <div style='border:2px solid #1f77b4;padding:12px;border-radius:12px;'>
            <h4>Affiliate Brokerage</h4>
            <p>Send your BUY/SELL signals to subscribers with your Binance link.</p>
            <a href="{BINANCE_AFFILIATE_URL}" target="_blank">Launch Binance Referral ➜</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    monetization_cols[1].markdown(
        f"""
        <div style='border:2px solid #2ca02c;padding:12px;border-radius:12px;'>
            <h4>Ecommerce Arbitrage</h4>
            <p>Source discounted products and list on Amazon with affiliate bump.</p>
            <a href="{AMAZON_AFFILIATE_URL}" target="_blank">Open Amazon Seller Central ➜</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    monetization_cols[2].markdown(
        f"""
        <div style='border:2px solid #ff7f0e;padding:12px;border-radius:12px;'>
            <h4>Info Product Funnel</h4>
            <p>Bundle exploding keywords into a paid mini-course.</p>
            <a href="{COURSE_AFFILIATE_URL}" target="_blank">Enroll & Earn Commission ➜</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Referral Growth Loop")
    if st.session_state.get("generated_referral"):
        referral_link = create_referral_link(st.session_state["generated_referral"])
        st.success(f"Share this link to earn 30% recurring: {referral_link}")
    else:
        st.info("Log in to generate your unique referral link and grow your lead base.")

    st.markdown("---")
    st.markdown(
        """
        <div style='margin-top:40px;padding:20px;background:#f5f5f5;border-radius:12px;'>
            <h4>Sponsored Slots</h4>
            <p>Scale your income with partners:</p>
            <ul>
                <li><a href="https://partners.cloudflare.com/" target="_blank">Cloudflare Affiliate</a> - secure SaaS clients.</li>
                <li><a href="https://www.bluehost.com/track/affiliate/" target="_blank">Bluehost Hosting Deals</a> - bundle with SEO packages.</li>
                <li><a href="https://wise.com/invite/" target="_blank">Wise Business Accounts</a> - collect referral payouts globally.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Disclaimer: No financial advice. Perform independent due diligence before acting on any signal.")


# ------------------------- APP ENTRY -------------------------
def main():
    initialize_state()
    init_db()
    sidebar_ui()
    main_content()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# Installation Command:
# pip install streamlit pandas yfinance pytrends plotly requests openpyxl
#
# Directory Tree After Setup:
# .
# ├── app.py
# └── growth_engine.db  (auto-created on first run)
#
# Run Instructions:
# 1. Execute: streamlit run app.py
# 2. Use sidebar to register, capture emails, and unlock premium tier.
# 3. Share referral link and promote affiliate offers displayed in-app to monetize.
