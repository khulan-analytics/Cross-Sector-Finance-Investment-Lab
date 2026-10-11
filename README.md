# Cross-Sector Finance & Investment Lab

A recruiter-facing Streamlit portfolio for finance, investment research, credit risk, and business analytics. It uses a focused sample of real public companies and does **not** require recruiters to provide an API key or upload data to see the default demonstrations.

## Modules
- **Overview:** project summary and selected company universe.
- **Company Screener:** available reported company fundamentals retrieved through Yahoo Finance using `yfinance`; values are shown as unavailable when the provider does not return them.
- **Financial Analysis & DCF:** a five-year-style FCFF scenario model using retrieved reported revenue and operating income as starting points. Growth, margins, tax, D&A, CapEx, working capital, WACC, and terminal growth are clearly labeled analyst assumptions. Includes WACC/terminal-growth sensitivity. Enterprise value is not equity value per share.
- **Investment & Portfolio Risk:** historical adjusted prices, returns, volatility, Sharpe ratio (risk-free rate assumed to be zero), maximum drawdown, and beta vs. SPY.
- **Stock Technical Analysis:** moving averages, Bollinger Bands, RSI, and MACD.
- **Credit Risk Modeling:** automatically loads a public loan-default dataset and evaluates Logistic Regression or Random Forest on a stratified held-out test set. Includes confusion matrix, ROC-AUC, precision, recall, F1, threshold controls, and optional CSV upload.
- **Business Intelligence:** automatically loads a public Sample Superstore transaction dataset and displays sales/profit KPIs, regional/category charts, monthly trends, and optional CSV upload.

## Run locally on Mac
```bash
cd ~/Downloads/cross_sector_finance_streamlit_portfolio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## API keys
No FMP key is required. Public market prices and available company fundamentals use Yahoo Finance through `yfinance`. The provider may change, throttle, or omit data; the app should show missing values rather than fabricate them. No recruiter should need to create an account or enter a key.

## Demo data sources
- Credit risk dataset CSV: https://raw.githubusercontent.com/Dhanshri-S/Credit_risk_model_using-logreg/main/credit_risk_dataset.csv
- Sample Superstore CSV: https://gist.githubusercontent.com/SharmaAshwini/8bd642f6c46792a9c40c8ccad60391e9/raw/Sample_Superstore.csv

The app caches public demo data for 24 hours and uses public internet access to retrieve it when not bundled locally. Upload controls are optional fallback/custom-data features.

## Data integrity and limitations
- The sample company universe is for research practice, not a recommendation list.
- Yahoo Finance data is obtained through an unofficial interface; financial-statement availability and labels can vary by company and over time. Check source periods and definitions before making comparisons.
- The DCF is an educational scenario, not investment advice. User-set assumptions are not reported facts. It produces enterprise value only and does not automatically calculate per-share equity value.
- A conventional industrial FCFF DCF is generally not the preferred primary valuation method for banks; the app warns when JPM or GS is selected.
- Credit-risk metrics are historical test-set results, not evidence of future lending performance or fairness.
- Superstore is a historical educational dataset, not current company financial reporting.

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `.gitignore`, `README.md`, and `data/company_universe.csv` to a public GitHub repository.
2. Go to https://share.streamlit.io/ and deploy the repository's `app.py`.
3. No secrets are required for the demo modules. Ensure the deployment environment has internet access for public data retrieval.

Do not publish private credentials or sensitive data. Check the data-provider terms before public deployment.
