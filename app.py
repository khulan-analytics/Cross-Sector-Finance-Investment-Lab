from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def download_prices(symbols, period="1y"):
    """
    Download real historical stock prices from Yahoo Finance.
    Returns a DataFrame with one adjusted-price column per ticker.
    """
    import pandas as pd
    import yfinance as yf

    symbols = list(symbols)

    if not symbols:
        raise ValueError("Please select at least one stock ticker.")

    data = yf.download(
        tickers=symbols,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True
    )

    if data is None or data.empty:
        raise ValueError(
            "No stock price data was returned. Check your internet "
            "connection, ticker symbols, and selected date range."
        )

    # Handle the multi-level columns returned by some yfinance versions.
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Close" in data.columns.get_level_values(1):
            prices = data.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("Yahoo Finance did not return closing prices.")
    else:
        # Handle a request for a single ticker.
        if "Close" not in data.columns:
            raise ValueError("Yahoo Finance did not return closing prices.")

        ticker = symbols[0]
        prices = data[["Close"]].rename(columns={"Close": ticker}).copy()

    # Ensure the result always has ticker names as column names.
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    prices = prices.dropna(axis=1, how="all")
    prices = prices.dropna(axis=0, how="all")

    if prices.empty:
        raise ValueError("No valid historical prices were available.")

    return prices

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Cross-Sector Finance & Investment Lab",
    page_icon="📊",
    layout="wide",
)

COMPANIES = {
    "MSFT": {"name": "Microsoft", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet", "sector": "Communication Services"},
    "JPM": {"name": "JPMorgan Chase", "sector": "Banking"},
    "GS": {"name": "Goldman Sachs", "sector": "Banking"},
    "LMT": {"name": "Lockheed Martin", "sector": "Aerospace & Defense"},
    "NOC": {"name": "Northrop Grumman", "sector": "Aerospace & Defense"},
    "RTX": {"name": "RTX", "sector": "Aerospace & Defense"},
    "LLY": {"name": "Eli Lilly", "sector": "Healthcare"},
    "COST": {"name": "Costco", "sector": "Consumer Staples"},
    "CAT": {"name": "Caterpillar", "sector": "Industrials"},
    "XOM": {"name": "Exxon Mobil", "sector": "Energy"},
}
TICKERS = list(COMPANIES.keys())
CREDIT_DATA_URL = "https://raw.githubusercontent.com/Dhanshri-S/Credit_risk_model_using-logreg/main/credit_risk_dataset.csv"
BI_DATA_URL = "https://gist.githubusercontent.com/SharmaAshwini/8bd642f6c46792a9c40c8ccad60391e9/raw/Sample_Superstore.csv"


# -------------------------------------------------------------------
# Helpers: data sources and calculations
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def company_fundamentals(symbol: str) -> dict:
    """Retrieve available reported company fundamentals through yfinance; no API key required."""
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.get_info()
    except Exception:
        info = {}
    try:
        income = ticker.financials
    except Exception:
        income = pd.DataFrame()
    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    def row_value(frame: pd.DataFrame, labels: tuple[str, ...]) -> float:
        if frame is None or frame.empty:
            return np.nan
        normalized = {str(idx).strip().lower(): idx for idx in frame.index}
        for label in labels:
            actual = normalized.get(label.lower())
            if actual is not None:
                values = pd.to_numeric(frame.loc[actual], errors="coerce").dropna()
                if not values.empty:
                    return float(values.iloc[0])
        return np.nan

    revenue = row_value(income, ("Total Revenue", "Operating Revenue"))
    operating_income = row_value(income, ("Operating Income", "EBIT"))
    net_income = row_value(income, ("Net Income", "Net Income Common Stockholders"))
    total_debt = row_value(balance, ("Total Debt", "Long Term Debt And Capital Lease Obligation"))
    cash = row_value(balance, ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"))
    fcf = row_value(cashflow, ("Free Cash Flow",))
    if not np.isfinite(fcf):
        ocf = row_value(cashflow, ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"))
        capex = row_value(cashflow, ("Capital Expenditure", "Capital Expenditures"))
        if np.isfinite(ocf) and np.isfinite(capex):
            fcf = ocf - abs(capex)

    dates = []
    if income is not None and not income.empty:
        dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in income.columns]
    revenue_growth = np.nan
    if income is not None and not income.empty and len(income.columns) >= 2:
        revenue_row = next((i for i in income.index if str(i).strip().lower() in {"total revenue", "operating revenue"}), None)
        if revenue_row is not None:
            vals = pd.to_numeric(income.loc[revenue_row], errors="coerce").dropna()
            if len(vals) >= 2 and vals.iloc[1] != 0:
                revenue_growth = vals.iloc[0] / vals.iloc[1] - 1

    return {
        "Ticker": symbol,
        "Company": info.get("longName") or info.get("shortName") or COMPANIES[symbol]["name"],
        "Sector": COMPANIES[symbol]["sector"],
        "Currency": info.get("currency", "USD"),
        "Fiscal period": dates[0] if dates else "Unavailable",
        "Market cap": info.get("marketCap", np.nan),
        "Revenue": revenue,
        "Revenue growth YoY": revenue_growth,
        "Operating income": operating_income,
        "Operating margin": operating_income / revenue if np.isfinite(operating_income) and np.isfinite(revenue) and revenue > 0 else np.nan,
        "Net income": net_income,
        "Net margin": net_income / revenue if np.isfinite(net_income) and np.isfinite(revenue) and revenue > 0 else np.nan,
        "Total debt": total_debt,
        "Cash and equivalents": cash,
        "Net debt": total_debt - cash if np.isfinite(total_debt) and np.isfinite(cash) else np.nan,
        "Free cash flow": fcf,
        "FCF margin": fcf / revenue if np.isfinite(fcf) and np.isfinite(revenue) and revenue > 0 else np.nan,
        "Data source": "Yahoo Finance via yfinance; latest available reported data",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def company_statement_tables(symbol: str) -> dict:
    """Return annual statements from Yahoo Finance, preserving the reported labels and periods."""
    ticker = yf.Ticker(symbol)
    result = {}
    for name, attr in [("income", "financials"), ("balance", "balance_sheet"), ("cashflow", "cashflow")]:
        try:
            frame = getattr(ticker, attr)
            if frame is None or frame.empty:
                result[name] = pd.DataFrame()
            else:
                table = frame.copy()
                table.columns = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in table.columns]
                result[name] = table
        except Exception:
            result[name] = pd.DataFrame()
    try:
        result["info"] = {k: v for k, v in ticker.get_info().items() if isinstance(v, (str, int, float, bool)) or v is None}
    except Exception:
        result["info"] = {}
    return result


@st.cache_data(ttl=86400, show_spinner=False)
def load_credit_demo_data() -> pd.DataFrame:
    """Load a real public credit-risk dataset; use a bundled local copy when available."""
    local_path = os.path.join(os.path.dirname(__file__), "data", "credit_risk_dataset.csv")
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        response = requests.get(CREDIT_DATA_URL, timeout=45, headers={"User-Agent": "CrossSectorFinanceLab/1.0"})
        response.raise_for_status()
        from io import BytesIO
        data = pd.read_csv(BytesIO(response.content))
    if "loan_status" not in data.columns:
        raise ValueError("The public credit dataset did not contain the expected loan_status target column.")
    return data


@st.cache_data(ttl=86400, show_spinner=False)
def load_bi_demo_data() -> pd.DataFrame:
    """Load a real public Sample Superstore transaction dataset; local copy takes priority."""
    local_path = os.path.join(os.path.dirname(__file__), "data", "Sample_Superstore.csv")
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        response = requests.get(BI_DATA_URL, timeout=45, headers={"User-Agent": "CrossSectorFinanceLab/1.0"})
        response.raise_for_status()
        from io import BytesIO
        data = pd.read_csv(BytesIO(response.content))
    # Normalize a few common field-name variants so the dashboard works with public mirrors.
    rename = {
        "Order Date": "Order Date", "OrderDate": "Order Date", "Order_Date": "Order Date",
        "Sales": "Sales", "Profit": "Profit", "Region": "Region", "Category": "Category",
    }
    data = data.rename(columns={c: rename[c] for c in data.columns if c in rename})
    return data


def fmt_money(value, currency="$"):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1e12:
        return f"{currency}{value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"{currency}{value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"{currency}{value / 1e6:,.2f}M"
    return f"{currency}{value:,.2f}"


def price_risk_metrics(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    output = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        r = returns[ticker].dropna() if ticker in returns else pd.Series(dtype=float)
        if series.empty or r.empty:
            continue
        drawdown = series / series.cummax() - 1
        annual_vol = r.std(ddof=1) * np.sqrt(252) if len(r) > 1 else np.nan
        annual_return = (series.iloc[-1] / series.iloc[0]) ** (252 / max(len(r), 1)) - 1 if len(r) else np.nan
        sharpe = (r.mean() * 252) / annual_vol if pd.notna(annual_vol) and annual_vol > 0 else np.nan
        beta = np.nan
        if benchmark in returns.columns and ticker != benchmark:
            joined = pd.concat([r.rename("asset"), returns[benchmark].dropna().rename("market")], axis=1).dropna()
            if len(joined) > 2 and joined["market"].var() > 0:
                beta = joined["asset"].cov(joined["market"]) / joined["market"].var()
        output.append({
            "Ticker": ticker,
            "Period return": series.iloc[-1] / series.iloc[0] - 1,
            "Annualized return (approx.)": annual_return,
            "Annualized volatility": annual_vol,
            "Sharpe ratio (rf=0 assumption)": sharpe,
            "Maximum drawdown": drawdown.min(),
            "Beta vs SPY": beta,
        })
    return pd.DataFrame(output)


def add_indicators(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": close.dropna()})
    df["SMA 20"] = df["Close"].rolling(20).mean()
    df["SMA 50"] = df["Close"].rolling(50).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["BB upper"] = df["SMA 20"] + 2 * rolling_std
    df["BB lower"] = df["SMA 20"] - 2 * rolling_std
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI 14"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df


def run_credit_risk(data: pd.DataFrame, target: str, model_name: str, test_size: float, threshold: float = 0.5):
    """Train/evaluate a baseline classifier using a held-out stratified test set."""
    if target not in data.columns:
        raise ValueError("Select a valid target column.")
    data = data.dropna(subset=[target]).copy()
    if data[target].nunique() != 2:
        raise ValueError("This module supports a binary target with exactly two classes.")
    y = data[target]
    X = data.drop(columns=[target])
    # Remove identifier-like columns to reduce accidental memorization.
    drop_cols = [c for c in X.columns if X[c].nunique(dropna=True) > 0.98 * len(X)]
    X = X.drop(columns=drop_cols, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain after excluding near-unique identifier columns.")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipe, numeric_cols), ("categorical", categorical_pipe, categorical_cols)])
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced") if model_name == "Logistic Regression" else RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    classes = list(model.classes_)
    if 1 in classes:
        positive_index = classes.index(1)
        positive_label = 1
    else:
        positive_index = len(classes) - 1
        positive_label = classes[positive_index]
    proba = model.predict_proba(X_test)[:, positive_index]
    preds = np.where(proba >= threshold, positive_label, classes[0] if classes[0] != positive_label else classes[1])
    auc = roc_auc_score((y_test == positive_label).astype(int), proba) if y_test.nunique() == 2 else np.nan
    report = classification_report(y_test, preds, labels=classes, zero_division=0, output_dict=True)
    return {
        "model": model, "data": data, "dropped_columns": drop_cols,
        "accuracy": accuracy_score(y_test, preds), "auc": auc,
        "confusion": confusion_matrix(y_test, preds, labels=classes), "classes": classes,
        "report": report, "test_count": len(y_test), "test_y": y_test, "preds": preds,
        "proba": proba, "features": X.columns.tolist(), "positive_label": positive_label,
        "threshold": threshold, "test_size": test_size, "model_name": model_name,
    }


# -------------------------------------------------------------------
# Sidebar navigation
# -------------------------------------------------------------------
st.title("Cross-Sector Finance & Investment Lab")
st.caption("Created by Khulan Ganzorig| MS in Business Analytics, University at Buffalo")
st.caption("A finance-oriented analytics portfolio: company screening, valuation, portfolio risk, technical analysis, and credit-risk modeling.")

with st.sidebar:
    st.header("Research menu")
    page = st.radio(
        "Choose a module",
        [
            "Overview",
            "Company Screener",
            "Financial Analysis & DCF",
            "Investment & Portfolio Risk",
            "Stock Technical Analysis",
            "Credit Risk Modeling",
            "Business Intelligence",
        ],
    )
    st.divider()
    st.caption("No API key required. Market prices and available fundamentals are retrieved from Yahoo Finance via yfinance. Data availability can vary by company and provider.")

# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------
if page == "Overview":
    st.subheader("Project overview")
    st.write(
        "This portfolio is designed to demonstrate practical skills for finance, banking, risk analytics, and business analytics roles. "
        "It uses a focused sample of real public companies instead of trying to cover every listed company."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Company sample", "12")
    c2.metric("Analytics modules", "6")
    c3.metric("Demo data policy", "No fabricated metrics")
    st.markdown("### What this project demonstrates")
    st.markdown(
        "- **Financial analysis:** reported revenue, operating margin, net income, debt, and free cash flow.\n"
        "- **Valuation:** an editable DCF model with assumptions clearly separated from reported data.\n"
        "- **Investment analysis:** historical returns, volatility, Sharpe ratio, drawdown, and beta.\n"
        "- **Technical analysis:** moving averages, Bollinger Bands, RSI, and MACD.\n"
        "- **Credit risk:** automatically load a public loan-default dataset, evaluate Logistic Regression or Random Forest, and inspect default-class metrics.\n"
        "- **Business intelligence:** automatically load public Sample Superstore transactions and explore sales/profit by category, region, and date."
    )
    st.markdown("### Company sample")
    st.dataframe(pd.DataFrame([
        {"Ticker": t, "Company": COMPANIES[t]["name"], "Sector": COMPANIES[t]["sector"]}
        for t in TICKERS
    ]), use_container_width=True, hide_index=True)
    st.info("No API key required. Company fundamentals and historical prices use Yahoo Finance via yfinance when available. Credit-risk and BI demo datasets load automatically from public CSV sources; uploads are optional.")

# -------------------------------------------------------------------
# Company screener
# -------------------------------------------------------------------
elif page == "Company Screener":
    st.subheader("Company Screener")
    st.write("Explore available reported fundamentals for a focused cross-sector company sample. No API key is required.")
    selected = st.multiselect("Companies", TICKERS, default=["MSFT", "JPM", "LMT", "NOC", "NVDA"])
    if st.button("Load company fundamentals", type="primary"):
        records, errors = [], []
        progress = st.progress(0)
        for i, ticker in enumerate(selected):
            try:
                records.append(company_fundamentals(ticker))
            except Exception as exc:
                errors.append({"Ticker": ticker, "Error": str(exc)})
            progress.progress((i + 1) / max(len(selected), 1))
        if records:
            st.session_state["fundamentals"] = pd.DataFrame(records)
        if errors:
            st.session_state["fundamentals_errors"] = pd.DataFrame(errors)
    if "fundamentals" in st.session_state:
        fundamentals = st.session_state["fundamentals"]
        view = fundamentals[fundamentals["Ticker"].isin(selected)]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download fundamentals CSV", view.to_csv(index=False).encode("utf-8"), file_name="company_fundamentals.csv", mime="text/csv")
        st.caption("Source: Yahoo Finance via yfinance. These are provider-reported fields where available; N/A means a value was not returned, not zero. Confirm reporting period, currency, and definitions before comparing companies.")
    if "fundamentals_errors" in st.session_state:
        with st.expander("Data retrieval issues"):
            st.dataframe(st.session_state["fundamentals_errors"], use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# Financial analysis and DCF
# -------------------------------------------------------------------
elif page == "Financial Analysis & DCF":
    st.subheader("Financial Analysis & DCF Valuation")
    st.write("A recruiter-ready valuation case using available reported company financials. No API key is required.")
    ticker = st.selectbox("Company", TICKERS, index=TICKERS.index("LMT"), key="dcf_ticker")
    st.caption("Data source: Yahoo Finance via yfinance. Financial statement coverage varies. Forecast inputs are analyst assumptions, not reported facts.")
    with st.spinner(f"Retrieving available reported statements for {ticker}..."):
        try:
            statements = company_statement_tables(ticker)
            income = statements.get("income", pd.DataFrame())
            cashflow = statements.get("cashflow", pd.DataFrame())
            info = statements.get("info", {})
        except Exception as exc:
            income, cashflow, info = pd.DataFrame(), pd.DataFrame(), {}
            st.error(f"Could not retrieve financial statements: {exc}")

    def find_row(frame: pd.DataFrame, labels: tuple[str, ...]) -> pd.Series:
        if frame is None or frame.empty:
            return pd.Series(dtype=float)
        lookup = {str(idx).strip().lower(): idx for idx in frame.index}
        for label in labels:
            idx = lookup.get(label.lower())
            if idx is not None:
                return pd.to_numeric(frame.loc[idx], errors="coerce").dropna()
        return pd.Series(dtype=float)

    revenue_series = find_row(income, ("Total Revenue", "Operating Revenue"))
    operating_series = find_row(income, ("Operating Income", "EBIT"))
    net_income_series = find_row(income, ("Net Income", "Net Income Common Stockholders"))
    if income.empty or revenue_series.empty or operating_series.empty:
        st.warning("The provider did not return enough reported income-statement data for this company to initialize a DCF. Try another company. The app does not substitute invented financial values.")
        if not income.empty:
            st.dataframe(income, use_container_width=True)
    else:
        revenue = float(revenue_series.iloc[0])
        operating_income = float(operating_series.iloc[0])
        reported_period = str(revenue_series.index[0])
        m1, m2, m3 = st.columns(3)
        m1.metric("Reported revenue", fmt_money(revenue))
        m2.metric("Reported operating income", fmt_money(operating_income))
        m3.metric("Reported net income", fmt_money(float(net_income_series.iloc[0])) if not net_income_series.empty else "N/A")
        st.caption(f"Latest returned annual statement period: {reported_period}. Values use the provider's reported currency, usually USD for this company.")
        with st.expander("View reported annual income statement"):
            st.dataframe(income.T.reset_index().rename(columns={"index": "Fiscal period"}), use_container_width=True, hide_index=True)
        if ticker in {"JPM", "GS"}:
            st.warning("Banks have different balance-sheet and cash-flow economics. A conventional industrial-company FCFF DCF is generally not the preferred primary valuation method for banks; use this only as a modeling demonstration and consider dividend discount or excess-return valuation instead.")

        st.markdown("### FCFF-based DCF scenario")
        st.write("Starting revenue and operating income come from the reported statement above. All forward-looking inputs below are user-selected assumptions.")
        margin_default = float(np.clip(operating_income / revenue * 100, 0, 50)) if revenue > 0 else 10.0
        growth = st.slider("Annual revenue growth assumption", -10.0, 25.0, 5.0, 0.5, key="dcf_growth") / 100
        margin = st.slider("Forecast operating margin assumption (%)", 0.0, 50.0, margin_default, 0.5, key="dcf_margin") / 100
        tax_rate = st.slider("Tax rate assumption (%)", 0.0, 40.0, 21.0, 0.5, key="dcf_tax") / 100
        da_pct = st.slider("D&A as % of revenue assumption", 0.0, 15.0, 3.0, 0.5, key="dcf_da") / 100
        capex_pct = st.slider("Capital expenditure as % of revenue assumption", 0.0, 20.0, 4.0, 0.5, key="dcf_capex") / 100
        nwc_pct = st.slider("Incremental working capital as % of revenue increase", 0.0, 20.0, 2.0, 0.5, key="dcf_nwc") / 100
        wacc = st.slider("WACC (%)", 4.0, 18.0, 9.0, 0.25, key="dcf_wacc") / 100
        terminal_growth = st.slider("Terminal growth rate (%)", 0.0, 4.0, 2.0, 0.25, key="dcf_terminal") / 100
        years = st.slider("Forecast horizon (years)", 3, 10, 5, key="dcf_years")
        if revenue <= 0:
            st.error("Reported revenue must be positive for this DCF setup.")
        elif terminal_growth >= wacc:
            st.error("Terminal growth must be below WACC.")
        else:
            rows, pv_cashflows = [], 0.0
            previous_revenue = revenue
            for year in range(1, years + 1):
                forecast_revenue = revenue * ((1 + growth) ** year)
                ebit = forecast_revenue * margin
                nopat = ebit * (1 - tax_rate)
                da = forecast_revenue * da_pct
                capex = forecast_revenue * capex_pct
                nwc_change = (forecast_revenue - previous_revenue) * nwc_pct
                fcff = nopat + da - capex - nwc_change
                pv = fcff / ((1 + wacc) ** year)
                pv_cashflows += pv
                rows.append({"Forecast year": year, "Revenue forecast": forecast_revenue, "EBIT": ebit, "NOPAT": nopat, "D&A assumption": da, "CapEx assumption": capex, "Change in NWC assumption": nwc_change, "FCFF": fcff, "PV of FCFF": pv})
                previous_revenue = forecast_revenue
            terminal_value = rows[-1]["FCFF"] * (1 + terminal_growth) / (wacc - terminal_growth)
            pv_terminal = terminal_value / ((1 + wacc) ** years)
            enterprise_value = pv_cashflows + pv_terminal
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            a, b, c = st.columns(3)
            a.metric("PV of forecast FCFF", fmt_money(pv_cashflows))
            b.metric("PV of terminal value", fmt_money(pv_terminal))
            c.metric("Illustrative enterprise value", fmt_money(enterprise_value))
            # WACC / terminal-growth sensitivity, with all assumptions held constant.
            sensitivity = []
            for wacc_pct in [max(4.0, wacc * 100 - 1), wacc * 100, min(18.0, wacc * 100 + 1)]:
                row = {"WACC": f"{wacc_pct:.2f}%"}
                for growth_pct in [max(0.0, terminal_growth * 100 - 0.5), terminal_growth * 100, min(4.0, terminal_growth * 100 + 0.5)]:
                    g = growth_pct / 100
                    w = wacc_pct / 100
                    if g >= w:
                        val = np.nan
                    else:
                        tv = rows[-1]["FCFF"] * (1 + g) / (w - g)
                        pv_t = tv / ((1 + w) ** years)
                        pv_forecast = sum(r["FCFF"] / ((1 + w) ** r["Forecast year"]) for r in rows)
                        val = pv_forecast + pv_t
                    row[f"Terminal growth {growth_pct:.2f}%"] = val
                sensitivity.append(row)
            st.markdown("#### Enterprise value sensitivity")
            st.dataframe(pd.DataFrame(sensitivity), use_container_width=True, hide_index=True)
            st.caption("Enterprise value is not equity value per share. This simplified DCF does not adjust for net debt, non-operating assets, diluted shares, or company-specific factors. D&A, CapEx, working capital, growth, tax, WACC, and terminal growth are explicit assumptions. This is an educational model, not investment advice.")

# -------------------------------------------------------------------
# Investment and portfolio risk
# -------------------------------------------------------------------
elif page == "Investment & Portfolio Risk":
    st.subheader("Investment & Portfolio Risk")
    st.write("Analyze real historical adjusted closing prices. The sample portfolio is equally weighted among the tickers you select.")
    tickers = st.multiselect("Portfolio constituents", TICKERS, default=["MSFT", "JPM", "LMT", "NVDA", "XOM"])
    period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y"], index=2)
    include_spy = st.checkbox("Include SPY as market benchmark", value=True)
    if st.button("Download prices and calculate risk", type="primary"):
        symbols = list(dict.fromkeys(tickers + (["SPY"] if include_spy else [])))
        if not tickers:
            st.warning("Select at least one company.")
        else:
            with st.spinner("Downloading historical prices..."):
                prices = download_prices(tuple(symbols), period=period)
            if prices.empty:
                st.error("No price data returned. Try a different period or check connectivity.")
            else:
                st.session_state["portfolio_prices"] = prices
                st.session_state["portfolio_tickers"] = tickers
                st.session_state["portfolio_period"] = period
    if "portfolio_prices" in st.session_state:
        prices = st.session_state["portfolio_prices"]
        selected_cols = [t for t in st.session_state.get("portfolio_tickers", []) if t in prices.columns]
        if selected_cols:
            asset_prices = prices[selected_cols].dropna(how="all")
            returns = asset_prices.pct_change().dropna(how="all")
            weights = np.repeat(1 / len(selected_cols), len(selected_cols))
            portfolio_returns = returns.fillna(0).dot(weights)
            growth = (1 + portfolio_returns).cumprod()
            annual_vol = portfolio_returns.std(ddof=1) * np.sqrt(252) if len(portfolio_returns) > 1 else np.nan
            annual_return = (growth.iloc[-1] ** (252 / len(portfolio_returns)) - 1) if len(portfolio_returns) else np.nan
            sharpe = (portfolio_returns.mean() * 252 / annual_vol) if pd.notna(annual_vol) and annual_vol > 0 else np.nan
            max_dd = (growth / growth.cummax() - 1).min()
            a, b, c, d = st.columns(4)
            a.metric("Portfolio period return", f"{(growth.iloc[-1] - 1):.2%}" if len(growth) else "N/A")
            b.metric("Annualized volatility", f"{annual_vol:.2%}" if pd.notna(annual_vol) else "N/A")
            c.metric("Sharpe ratio (rf=0)", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
            d.metric("Maximum drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "N/A")
            normalized = asset_prices / asset_prices.iloc[0] * 100
            fig = px.line(normalized, title="Normalized price performance (start = 100)")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Individual asset risk metrics")
            risk = price_risk_metrics(prices, benchmark="SPY")
            risk = risk[risk["Ticker"].isin(selected_cols)]
            st.dataframe(risk, use_container_width=True, hide_index=True)
            st.download_button("Download risk metrics CSV", risk.to_csv(index=False).encode(), "portfolio_risk_metrics.csv", "text/csv")
            st.caption("Sharpe ratio assumes a 0% risk-free rate for simplicity. Annualized return is an approximation based on observed daily observations. Historical performance does not predict future returns.")

# -------------------------------------------------------------------
# Technical analysis
# -------------------------------------------------------------------
elif page == "Stock Technical Analysis":
    st.subheader("Stock Technical Analysis")
    ticker = st.selectbox("Ticker", TICKERS, index=TICKERS.index("LMT"))
    period = st.selectbox("Price history", ["6mo", "1y", "2y", "5y"], index=2, key="ta_period")
    if st.button("Load chart and indicators", type="primary"):
        with st.spinner("Downloading price history..."):
            prices = download_prices((ticker,), period=period)
        if prices.empty or ticker not in prices:
            st.error("Price history was unavailable.")
        else:
            st.session_state["ta_data"] = add_indicators(prices[ticker])
            st.session_state["ta_ticker"] = ticker
    if "ta_data" in st.session_state and st.session_state.get("ta_ticker") == ticker:
        data = st.session_state["ta_data"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Adjusted close"))
        for col in ["SMA 20", "SMA 50", "BB upper", "BB lower"]:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col, line={"dash": "dot"}))
        fig.update_layout(title=f"{ticker}: price and technical indicators", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        rsi_fig = px.line(data, y="RSI 14", title="RSI (14)")
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        st.plotly_chart(rsi_fig, use_container_width=True)
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=data.index, y=data["MACD signal"], name="Signal"))
        macd_fig.update_layout(title="MACD (12, 26, 9)")
        st.plotly_chart(macd_fig, use_container_width=True)
        st.caption("Technical indicators describe historical price behavior. They are not guaranteed trading signals.")

# -------------------------------------------------------------------
# Credit risk
# -------------------------------------------------------------------
elif page == "Credit Risk Modeling":
    st.subheader("Credit Risk Modeling")
    st.write("Explore a real public loan-default dataset. The dashboard loads data automatically; uploading your own dataset is optional.")
    st.caption("Dataset source: public Credit Risk Dataset mirror. Target: loan_status (1 = default, 0 = no default). Educational analysis only—not a lending decision system.")

    with st.spinner("Loading the public credit-risk dataset..."):
        try:
            demo_credit = load_credit_demo_data()
            data_source = "Public credit-risk demo dataset"
        except Exception as exc:
            demo_credit = None
            st.error(f"Could not load the automatic demo dataset: {exc}")
            st.info("If the data source is temporarily unavailable, upload the CSV below as a fallback.")

    with st.expander("Advanced: use your own credit-risk CSV (optional)"):
        uploaded_credit = st.file_uploader("Upload loan dataset CSV", type=["csv"], key="credit_upload")
        if uploaded_credit is not None:
            try:
                demo_credit = pd.read_csv(uploaded_credit)
                data_source = f"Uploaded file: {uploaded_credit.name}"
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")

    if demo_credit is not None:
        if "loan_status" not in demo_credit.columns:
            st.error("This dataset must contain a loan_status column (0 = no default, 1 = default).")
        else:
            df_credit = demo_credit.copy()
            total_rows = len(df_credit)
            default_rate = pd.to_numeric(df_credit["loan_status"], errors="coerce").mean()
            default_count = int((df_credit["loan_status"] == 1).sum())
            no_default_count = int((df_credit["loan_status"] == 0).sum())
            st.caption(f"Data source: {data_source} · {total_rows:,} rows · Missing values are imputed inside the training pipeline.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Loan records", f"{total_rows:,}")
            k2.metric("Historical defaults", f"{default_count:,}")
            k3.metric("Non-defaults", f"{no_default_count:,}")
            k4.metric("Default rate", f"{default_rate:.1%}" if pd.notna(default_rate) else "N/A")

            with st.expander("Preview dataset and data quality"):
                st.dataframe(df_credit.head(12), use_container_width=True, hide_index=True)
                missing = df_credit.isna().sum().rename("Missing values").to_frame()
                missing = missing[missing["Missing values"] > 0].sort_values("Missing values", ascending=False)
                if not missing.empty:
                    st.write("Missing values by column")
                    st.dataframe(missing, use_container_width=True)
                else:
                    st.success("No missing values detected in the current data.")

            c1, c2, c3 = st.columns(3)
            model_name = c1.selectbox("Model", ["Logistic Regression", "Random Forest"], key="credit_model_choice")
            test_size = c2.select_slider("Held-out test set", options=[0.2, 0.25, 0.3, 0.35, 0.4], value=0.3, format_func=lambda x: f"{x:.0%}", key="credit_test_size")
            threshold = c3.slider("Default classification threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05, key="credit_threshold")
            signature = (data_source, model_name, float(test_size), float(threshold), len(df_credit))
            if st.session_state.get("credit_signature") != signature or "credit_result" not in st.session_state:
                with st.spinner("Training model and evaluating held-out test records..."):
                    try:
                        result = run_credit_risk(df_credit, "loan_status", model_name, float(test_size), float(threshold))
                        st.session_state["credit_result"] = result
                        st.session_state["credit_signature"] = signature
                    except Exception as exc:
                        st.error(f"Could not train model: {exc}")
                        result = None
            else:
                result = st.session_state.get("credit_result")

            if result is not None:
                report = result["report"]
                positive_key = str(result["positive_label"])
                positive_report = report.get(positive_key, {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Test accuracy", f"{result['accuracy']:.1%}")
                m2.metric("ROC-AUC", f"{result['auc']:.3f}" if pd.notna(result["auc"]) else "N/A")
                m3.metric("Default recall", f"{positive_report.get('recall', 0):.1%}")
                m4.metric("Default precision", f"{positive_report.get('precision', 0):.1%}")

                left, right = st.columns([1, 1])
                with left:
                    cm = result["confusion"]
                    labels = [f"Class {x}" for x in result["classes"]]
                    fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues",
                                       labels={"x": "Predicted", "y": "Actual", "color": "Records"}, title="Confusion matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                with right:
                    st.markdown("**Default class (1) performance**")
                    st.write(f"- Precision: {positive_report.get('precision', 0):.1%}")
                    st.write(f"- Recall: {positive_report.get('recall', 0):.1%}")
                    st.write(f"- F1-score: {positive_report.get('f1-score', 0):.1%}")
                    st.write(f"- Threshold: {result['threshold']:.2f}")
                    st.write(f"- Held-out test records: {result['test_count']:,}")
                    st.caption("Higher recall catches more defaults but can increase false alarms. Threshold changes trade precision against recall.")

                st.markdown("#### Classification report")
                report_df = pd.DataFrame(result["report"]).transpose()
                st.dataframe(report_df, use_container_width=True)
                st.download_button("Download test-set predictions",
                    pd.DataFrame({"actual": result["test_y"].to_numpy(), "predicted": result["preds"], "predicted_default_probability": result["proba"]}).to_csv(index=False).encode("utf-8"),
                    file_name="credit_risk_test_predictions.csv", mime="text/csv")
                st.caption("Methodology: stratified train/test split; preprocessing is fit on training data only through a scikit-learn Pipeline. Results depend on this historical dataset and are not proof of future performance or fairness.")

# -------------------------------------------------------------------
# Business intelligence
# -------------------------------------------------------------------
elif page == "Business Intelligence":
    st.subheader("Business Intelligence: Sales & Profitability")
    st.write("The dashboard opens with a public Sample Superstore transaction dataset. Uploading another CSV is optional.")
    st.caption("Sample Superstore is a historical educational retail dataset, not current company financial reporting.")

    with st.spinner("Loading the public Sample Superstore dataset..."):
        try:
            bi_demo = load_bi_demo_data()
            bi_source = "Public Sample Superstore dataset"
        except Exception as exc:
            bi_demo = None
            st.error(f"Could not load the automatic BI demo dataset: {exc}")
            st.info("If the data source is temporarily unavailable, upload a CSV below as a fallback.")

    with st.expander("Advanced: use your own business CSV (optional)"):
        uploaded_bi = st.file_uploader("Upload business CSV", type=["csv"], key="bi_upload")
        if uploaded_bi is not None:
            try:
                bi_demo = pd.read_csv(uploaded_bi)
                bi_source = f"Uploaded file: {uploaded_bi.name}"
            except Exception as exc:
                st.error(f"Could not read uploaded CSV: {exc}")

    if bi_demo is not None:
        data = bi_demo.copy()
        st.caption(f"Data source: {bi_source} · {len(data):,} transaction rows")
        # Standard Superstore columns are expected; the generic controls below still work if columns differ.
        sales_col = next((c for c in data.columns if c.lower().strip() == "sales"), None)
        profit_col = next((c for c in data.columns if c.lower().strip() == "profit"), None)
        region_col = next((c for c in data.columns if c.lower().strip() == "region"), None)
        category_col = next((c for c in data.columns if c.lower().strip() == "category"), None)
        date_col = next((c for c in data.columns if c.lower().strip() in {"order date", "orderdate", "order_date"}), None)

        if sales_col and profit_col:
            data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")
            data[profit_col] = pd.to_numeric(data[profit_col], errors="coerce")
            total_sales = data[sales_col].sum()
            total_profit = data[profit_col].sum()
            margin = total_profit / total_sales if total_sales else np.nan
            a, b, c, d = st.columns(4)
            a.metric("Total sales", f"${total_sales:,.0f}")
            b.metric("Total profit", f"${total_profit:,.0f}")
            c.metric("Profit margin", f"{margin:.1%}" if pd.notna(margin) else "N/A")
            d.metric("Transaction lines", f"{len(data):,}")

            chart_cols = st.columns(2)
            if region_col:
                region_summary = data.groupby(region_col, dropna=False).agg(Sales=(sales_col, "sum"), Profit=(profit_col, "sum")).reset_index().sort_values("Sales", ascending=False)
                with chart_cols[0]:
                    st.plotly_chart(px.bar(region_summary, x=region_col, y="Sales", title="Sales by region"), use_container_width=True)
                with chart_cols[1]:
                    st.plotly_chart(px.bar(region_summary, x=region_col, y="Profit", title="Profit by region"), use_container_width=True)
            if category_col:
                category_summary = data.groupby(category_col, dropna=False).agg(Sales=(sales_col, "sum"), Profit=(profit_col, "sum")).reset_index().sort_values("Profit", ascending=False)
                st.plotly_chart(px.bar(category_summary, x=category_col, y="Profit", title="Profit by product category"), use_container_width=True)
                st.dataframe(category_summary, use_container_width=True, hide_index=True)
            if date_col:
                parsed_dates = pd.to_datetime(data[date_col], errors="coerce", format="mixed", dayfirst=False)
                if parsed_dates.notna().any():
                    monthly = data.assign(_month=parsed_dates.dt.to_period("M").astype(str)).groupby("_month", dropna=True)[sales_col].sum().reset_index()
                    monthly = monthly.rename(columns={"_month": "Month", sales_col: "Sales"})
                    st.plotly_chart(px.line(monthly, x="Month", y="Sales", markers=True, title="Monthly sales trend"), use_container_width=True)

        with st.expander("Explore another measure and dimension"):
            numeric = data.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in data.columns if c not in numeric]
            if numeric:
                measure = st.selectbox("Numeric measure", numeric, key="bi_measure")
                dimension = st.selectbox("Group by dimension", ["(none)"] + categorical, key="bi_dimension")
                if dimension == "(none)":
                    x1, x2 = st.columns(2)
                    x1.metric(f"Total {measure}", f"{data[measure].sum(skipna=True):,.2f}")
                    x2.metric(f"Average {measure}", f"{data[measure].mean(skipna=True):,.2f}")
                else:
                    grouped = data.groupby(dimension, dropna=False)[measure].sum().reset_index().sort_values(measure, ascending=False)
                    st.plotly_chart(px.bar(grouped, x=dimension, y=measure, title=f"{measure} by {dimension}"), use_container_width=True)
                    st.dataframe(grouped, use_container_width=True, hide_index=True)
            else:
                st.warning("No numeric columns found in this dataset.")
        st.download_button("Download current BI dataset", data.to_csv(index=False).encode("utf-8"), "business_data.csv", "text/csv")

st.divider()
st.caption(
    "Sources: Yahoo Finance via yfinance for available reported fundamentals and historical adjusted prices; public Credit Risk Dataset and Sample Superstore CSV sources for demo modules. "
    "Check provider plan permissions, data definitions, and redistribution terms before public deployment. Educational project—not investment advice."
)
