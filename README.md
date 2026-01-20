# Fast Market Dashboard

A market stress monitoring dashboard that produces composite risk indicators during periods of elevated volatility. Built for asset management risk teams.

## Overview

This dashboard aggregates multiple market stress signals into a single composite score (0-100) that indicates the current market regime:

| Score | Regime | Interpretation |
|-------|--------|----------------|
| 0-30 | Calm | Risk-on conditions |
| 30-50 | Normal | Standard monitoring |
| 50-70 | Elevated | Increased vigilance |
| 70-100 | Fast Market | Active risk management |

## Key Features

- **Multi-source data aggregation**: FRED, Yahoo Finance, Alpha Vantage
- **Backtested indicator weights**: Signal ratios derived from historical stress events
- **Non-linear composite models**: Linear, convex, convergence, and hybrid approaches
- **Dual output formats**: Interactive Streamlit dashboard and static HTML export
- **Historical analysis**: 5+ years of data with key event annotations

## Indicators

The composite score is built from 9 indicators across multiple asset classes:

| Tier | Indicator | Weight | Signal Ratio |
|------|-----------|--------|--------------|
| 1 | High Yield Credit Spread | 20% | 2.19x |
| 1 | BBB Credit Spread | 15% | 2.18x |
| 2 | VIX (Volatility) | 15% | 1.89x |
| 2 | Defensive Rotation (XLU/XLY) | 10% | 1.89x |
| 2 | USD Flight to Safety | 10% | 1.73x |
| 3 | Sector Correlation | 10% | 1.75x |
| 3 | VIX Term Structure | 10% | 1.63x |
| 4 | Safe Haven (GLD/SPY) | 5% | 1.38x |
| 4 | S&P 500 Drawdown | 5% | N/A |

*Signal Ratio = Average signal during stress / Average signal during calm periods*

## Backtest Performance

Tested against major market stress events:

| Event | Peak Score | Lead Days | Detected |
|-------|------------|-----------|----------|
| Fed Tightening 2018 | 92.1 | 0 | Yes |
| COVID Crash 2020 | 96.1 | 0 | Yes |
| Rate Shock 2022 | 93.5 | 0 | Yes |
| Tariff Shock 2025 | 88.6 | 4 | Yes |

**Detection Rate: 100%** (all events in data range)

## Installation

```bash
# Clone the repository
git clone https://github.com/ctjohn90/fast-market-dashboard.git
cd fast-market-dashboard

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
#   FRED_API_KEY=your_key_here
#   ALPHA_VANTAGE_API_KEY=your_key_here (optional)
#   DASHBOARD_PASSWORD=your_password_here
```

## Usage

### Streamlit Dashboard (Interactive)

```bash
streamlit run fast_market_dashboard/ui/dashboard.py
```

### HTML Export (Static/Shareable)

```bash
python -m fast_market_dashboard.ui.html_exporter
# Output: dist/index.html
```

### Run Backtest

```bash
python -m fast_market_dashboard.indicators.backtest
```

### Evaluate Individual Indicators

```bash
python -m fast_market_dashboard.indicators.indicator_eval
```

## Project Structure

```
fast_market_dashboard/
    config/             # Settings and API key management
    data/               # Data fetchers (FRED, Yahoo, Alpha Vantage) and cache
    indicators/         # Composite score calculation and backtesting
    models/             # Data models (dataclasses)
    ui/                 # Streamlit dashboard and HTML exporter
cache/
    market_data.db      # SQLite cache of fetched data
dist/
    index.html          # Exported static dashboard
```

## Data Sources

### Primary: FRED (Federal Reserve Economic Data)
- Free API, reliable, no aggressive rate limits
- Get API key: https://fred.stlouisfed.org/docs/api/api_key.html

### Secondary: Yahoo Finance
- Equity and ETF data for derived indicators
- Unofficial API (via yfinance)

### Optional: Alpha Vantage
- Technical indicators (RSI, MACD, ADX)
- Free tier: 25 calls/day

## Non-Linear Models

The dashboard supports multiple composite score models:

| Model | Description | Use Case |
|-------|-------------|----------|
| Linear | Simple weighted average | Daily monitoring (baseline) |
| Convex | Power transformation amplifies extremes | When stress is building |
| Convergence | Multiplier when multiple indicators elevated | Systemic risk detection |
| Hybrid | Combines convex + convergence + regime weights | Full signal extraction |

**Recommended approach**: Use Linear for daily monitoring. When Linear crosses 50, check Hybrid for amplified signal.

## Tech Stack

- **Python 3.11+**
- **pandas**: Data manipulation
- **Streamlit**: Interactive dashboard
- **Plotly**: Charting
- **SQLite**: Local data cache
- **httpx/fredapi/yfinance**: Data fetching

## License

Personal project / proof of concept.

## Acknowledgments

Inspired by institutional risk monitoring workflows and the need to synthesize multiple market signals into actionable regime indicators.
