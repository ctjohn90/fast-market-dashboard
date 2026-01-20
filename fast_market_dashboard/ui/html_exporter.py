"""Export dashboard as self-contained HTML with tabs."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from fast_market_dashboard.config import Settings
from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.indicators.calculator import IndicatorCalculator, WEIGHTS, INDICATOR_INFO


# Market events for annotations
MARKET_EVENTS = [
    {"date": "2025-01-27", "label": "DeepSeek shock", "color": "#ef4444"},
    {"date": "2025-04-02", "label": "Liberation Day tariffs", "color": "#ef4444"},
    {"date": "2025-04-09", "label": "90-day tariff pause", "color": "#10b981"},
    {"date": "2025-06-27", "label": "New all-time high", "color": "#10b981"},
    {"date": "2025-11-18", "label": "VIX spike to 52", "color": "#ef4444"},
    {"date": "2025-12-10", "label": "Fed hawkish pivot", "color": "#f59e0b"},
]


def hash_password(password: str) -> str:
    """Create SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_regime(score: float) -> dict:
    """Get regime info for a given score."""
    if score < 30:
        return {"color": "#10b981", "label": "CALM", "action": "Risk-on conditions", "class": "calm"}
    elif score < 50:
        return {"color": "#f59e0b", "label": "NORMAL", "action": "Standard monitoring", "class": "normal"}
    elif score < 70:
        return {"color": "#f97316", "label": "ELEVATED", "action": "Increased vigilance", "class": "elevated"}
    else:
        return {"color": "#ef4444", "label": "FAST MARKET", "action": "Active risk management", "class": "fast-market"}


def format_value(name: str, val: float) -> str:
    """Format indicator value based on type."""
    if "Spread" in name or "Yield" in name:
        return f"{val:.2f}%"
    elif "VIX" in name and "Term" not in name:
        return f"{val:.1f}"
    elif "Term Structure" in name:
        status = "backwardation" if val < 1 else "contango"
        return f"{val:.3f} ({status})"
    elif "USD" in name:
        return f"{val:.1f}"
    elif "Drawdown" in name:
        return f"{val:.0f}"
    elif "Correlation" in name:
        return f"{val:.3f}"
    elif "/" in name or "Rotation" in name:
        return f"{val:.3f}"
    else:
        return f"{val:.2f}"


def generate_alerts(result, history) -> list[tuple[str, str]]:
    """Generate active alerts based on thresholds."""
    alerts = []
    
    for key, ind in result.indicators.items():
        if ind.percentile >= 90:
            alerts.append(("HIGH", f"{ind.name} at {ind.percentile:.0f}th percentile"))
        elif ind.percentile >= 80:
            alerts.append(("ELEVATED", f"{ind.name} at {ind.percentile:.0f}th percentile"))
    
    if not history.empty and len(history) >= 5:
        change_5d = result.composite_score - history["composite"].iloc[-5]
        if change_5d >= 20:
            alerts.append(("HIGH", f"Composite up {change_5d:.0f} pts in 5 days"))
        elif change_5d >= 15:
            alerts.append(("ELEVATED", f"Composite up {change_5d:.0f} pts in 5 days"))
    
    if result.composite_score >= 70:
        alerts.append(("HIGH", "Fast market conditions active"))
    elif result.composite_score >= 50:
        alerts.append(("ELEVATED", "Elevated stress regime"))
    
    return alerts[:5]


def build_indicators_tab(result) -> str:
    """Build the Indicators reference tab HTML."""
    cards = []
    for key, info in INDICATOR_INFO.items():
        weight = WEIGHTS.get(key, 0) * 100
        current = result.indicators.get(key)
        
        if current:
            pct = current.percentile
            if pct >= 70:
                status_color = "#ef4444"
                status = "Elevated"
            elif pct >= 50:
                status_color = "#f59e0b"
                status = "Normal"
            else:
                status_color = "#10b981"
                status = "Low"
            pct_display = f"{pct:.0f}"
        else:
            status_color = "#6b7280"
            status = "No Data"
            pct_display = "N/A"
        
        cards.append(f'''
            <div class="indicator-card">
                <div class="indicator-card-header">
                    <div>
                        <h3 class="indicator-card-title">{info['name']}</h3>
                        <div class="indicator-card-source">{info['source']}</div>
                    </div>
                    <div class="indicator-card-status-wrap">
                        <div class="indicator-status" style="background: {status_color}22; border-color: {status_color}; color: {status_color};">
                            {status}: {pct_display}
                        </div>
                        <div class="indicator-weight">Weight: {weight:.0f}%</div>
                    </div>
                </div>
                <p class="indicator-description">{info['description']}</p>
                <div class="indicator-details">
                    <div class="indicator-detail-box">
                        <div class="indicator-detail-label">Interpretation</div>
                        <div class="indicator-detail-value">{info['interpretation']}</div>
                    </div>
                    <div class="indicator-detail-box">
                        <div class="indicator-detail-label">Backtest Performance</div>
                        <div class="indicator-detail-value">Signal Ratio: {info['signal_ratio']} | Detection: {info['detection_rate']}</div>
                    </div>
                </div>
            </div>
        ''')
    
    return f'''
        <h2 class="tab-heading">Indicator Reference Guide</h2>
        <p class="tab-intro">Each indicator in the composite score is selected based on backtesting against historical stress events.</p>
        {"".join(cards)}
    '''


def build_backtest_tab() -> str:
    """Build the Backtest results tab HTML."""
    return '''
        <h2 class="tab-heading">Backtesting Methodology & Results</h2>
        
        <div class="section-box">
            <h3 style="color: #f1f5f9; margin-bottom: 0.75rem;">How We Test Indicators</h3>
            <p style="color: #94a3b8;">Each indicator is evaluated against <strong style="color: #e2e8f0;">6 major market stress events</strong>:</p>
            <ul class="stress-events">
                <li><strong>2008-09:</strong> Financial Crisis (56.8% drawdown)</li>
                <li><strong>2011:</strong> Debt Ceiling Crisis (19.4% drawdown)</li>
                <li><strong>2015:</strong> China Devaluation (12.4% drawdown)</li>
                <li><strong>2018:</strong> Fed Tightening (19.8% drawdown)</li>
                <li><strong>2020:</strong> COVID Crash (33.9% drawdown)</li>
                <li><strong>2022:</strong> Rate Shock (25.4% drawdown)</li>
            </ul>
        </div>
        
        <div class="metrics-grid">
            <div class="section-box">
                <h4 style="color: #f1f5f9; margin-bottom: 0.5rem;">Signal Ratio</h4>
                <p style="color: #94a3b8; font-size: 0.85rem;">Average percentile during stress / Average during normal times.</p>
                <p style="color: #10b981; font-size: 0.85rem; margin-top: 0.5rem;">Higher is better. 2.0x = twice as elevated during stress.</p>
            </div>
            <div class="section-box">
                <h4 style="color: #f1f5f9; margin-bottom: 0.5rem;">Detection Rate</h4>
                <p style="color: #94a3b8; font-size: 0.85rem;">% of stress events where indicator exceeded 70th percentile.</p>
                <p style="color: #10b981; font-size: 0.85rem; margin-top: 0.5rem;">Higher is better. 100% = caught every major event.</p>
            </div>
        </div>
        
        <h3 class="section-title">Indicator Rankings</h3>
        <table class="rankings-table">
            <thead><tr><th>Indicator</th><th>Signal Ratio</th><th>Detection</th><th>Tier</th></tr></thead>
            <tbody>
                <tr><td>HY Credit Spread</td><td>2.19x</td><td>100%</td><td>1</td></tr>
                <tr><td>BBB Credit Spread</td><td>2.18x</td><td>100%</td><td>1</td></tr>
                <tr><td>VIX</td><td>1.89x</td><td>100%</td><td>2</td></tr>
                <tr><td>Defensive Rotation</td><td>1.89x</td><td>100%</td><td>2</td></tr>
                <tr><td>USD Index</td><td>1.73x</td><td>100%</td><td>2</td></tr>
                <tr><td>Sector Correlation</td><td>1.75x</td><td>100%</td><td>3</td></tr>
                <tr><td>VIX Term Structure</td><td>1.63x</td><td>100%</td><td>3</td></tr>
                <tr><td>Safe Haven (GLD/SPY)</td><td>1.38x</td><td>100%</td><td>4</td></tr>
                <tr><td>S&P 500 Drawdown</td><td>N/A</td><td>100%</td><td>4</td></tr>
            </tbody>
        </table>
        
        <h3 class="section-title">What Didn\'t Work</h3>
        <table class="rankings-table">
            <thead><tr><th>Indicator</th><th>Signal Ratio</th><th>Problem</th></tr></thead>
            <tbody>
                <tr><td class="failed">RSI (SPY, QQQ, IWM)</td><td class="failed">0.53-0.58x</td><td>Goes DOWN during stress</td></tr>
                <tr><td class="failed">10Y-2Y Spread</td><td class="failed">0.61x</td><td>Only 33% detection - too slow</td></tr>
                <tr><td class="failed">SKEW Index</td><td class="failed">0.22x</td><td>Opposite of expected</td></tr>
                <tr><td class="warning">Fed Funds Rate</td><td class="warning">1.02x</td><td>Policy-driven, not market</td></tr>
            </tbody>
        </table>
    '''


def build_models_tab() -> str:
    """Build the Models (non-linear analysis) tab HTML."""
    return '''
        <h2 class="tab-heading">Linear vs Non-Linear Models</h2>
        <p class="tab-intro">Testing whether non-linear relationships improve fast market detection.</p>
        
        <div class="section-box">
            <h4 style="color: #ef4444; margin-bottom: 0.75rem;">What Linear Models Miss</h4>
            <ul style="color: #94a3b8; padding-left: 1.25rem;">
                <li style="margin-bottom: 0.5rem;"><strong style="color: #e2e8f0;">Convexity at extremes</strong> - Going from 70th to 85th percentile is fundamentally different than 40th to 55th.</li>
                <li style="margin-bottom: 0.5rem;"><strong style="color: #e2e8f0;">Convergence effects</strong> - When VIX, credit, and USD all spike together, it\'s multiplicative, not additive.</li>
                <li style="margin-bottom: 0.5rem;"><strong style="color: #e2e8f0;">Regime shifts</strong> - In calm markets, credit leads. In stress, everything correlates.</li>
                <li><strong style="color: #e2e8f0;">Lead-lag relationships</strong> - Credit spreads widen before equities fall.</li>
            </ul>
        </div>
        
        <h3 class="section-title">Backtest Results: 6 Major Stress Events</h3>
        <table class="rankings-table">
            <thead><tr><th>Event</th><th>Linear</th><th>Hybrid</th></tr></thead>
            <tbody>
                <tr><td>Financial Crisis (2008)</td><td>72</td><td style="color: #10b981;">95</td></tr>
                <tr><td>Debt Ceiling (2011)</td><td>72</td><td style="color: #10b981;">95</td></tr>
                <tr><td>China Devaluation (2015)</td><td>72</td><td style="color: #10b981;">95</td></tr>
                <tr><td>Fed Tightening (2018)</td><td>71</td><td style="color: #10b981;">93</td></tr>
                <tr><td>COVID Crash (2020)</td><td>73</td><td style="color: #10b981;">95</td></tr>
                <tr><td>Rate Shock (2022)</td><td>72</td><td style="color: #10b981;">99</td></tr>
            </tbody>
        </table>
        
        <h3 class="section-title">Model Performance</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Signal Ratio</div>
                <div class="metric-value" style="color: #3b82f6;">+24%</div>
                <div class="metric-sublabel">Hybrid vs Linear</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Peak Score</div>
                <div class="metric-value" style="color: #10b981;">95-99</div>
                <div class="metric-sublabel">Hybrid in stress</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Lead Time</div>
                <div class="metric-value" style="color: #f59e0b;">+0.8d</div>
                <div class="metric-sublabel">Early warning</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Precision</div>
                <div class="metric-value" style="color: #ef4444;">-28%</div>
                <div class="metric-sublabel">More false positives</div>
            </div>
        </div>
        
        <h3 class="section-title">Practical Recommendation</h3>
        <div class="section-box" style="background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%); border-color: #3b82f6;">
            <h4 style="color: #f1f5f9; margin-bottom: 1rem;">Two-Stage Monitoring Approach</h4>
            <div class="stage-row">
                <span class="stage-badge" style="background: #10b981;">STAGE 1</span>
                <span class="stage-text">Use <strong>Linear model</strong> for daily monitoring. Fewer false alarms, stable baseline.</span>
            </div>
            <div class="stage-row">
                <span class="stage-badge" style="background: #f59e0b;">STAGE 2</span>
                <span class="stage-text">When Linear crosses <strong>50+</strong>, switch to <strong>Hybrid lens</strong> for amplified signal.</span>
            </div>
            <div class="stage-row">
                <span class="stage-badge" style="background: #ef4444;">STAGE 3</span>
                <span class="stage-text">If Hybrid exceeds <strong>85</strong>, treat as confirmed fast market.</span>
            </div>
        </div>
    '''


def export_html(output_path: Path | str | None = None, password: str | None = None) -> Path:
    """
    Generate self-contained HTML dashboard with tabs.
    
    Args:
        output_path: Where to save the HTML file. Defaults to dist/index.html
        password: If provided, adds client-side password protection
        
    Returns:
        Path to the generated file
    """
    # Get password from env if not provided
    if password is None:
        password = os.getenv("DASHBOARD_PASSWORD")
    
    password_hash = hash_password(password) if password else None
    
    settings = Settings()
    calculator = IndicatorCalculator(settings)
    
    # Get current state
    result = calculator.calculate()
    if result is None:
        raise ValueError("No data available. Run the FRED fetcher first.")
    
    # Get history for chart (get 1 year for period selector options)
    history = calculator.get_history(days=365)
    
    # Calculate 5-day delta
    if not history.empty and len(history) >= 5:
        prev_score = history["composite"].iloc[-5]
        delta = result.composite_score - prev_score
        if abs(delta) < 0.5:
            delta_str, delta_color = "0", "#6b7280"
        elif delta > 0:
            delta_str, delta_color = f"+{delta:.1f}", "#ef4444"
        else:
            delta_str, delta_color = f"{delta:.1f}", "#10b981"
    else:
        delta_str, delta_color = "N/A", "#6b7280"
    
    # Prepare chart data (full year)
    chart_dates = []
    chart_values = []
    sp500_returns = []
    if not history.empty:
        chart_dates = [d.strftime("%Y-%m-%d") for d in history.index]
        chart_values = [round(v, 1) for v in history["composite"].values]
        
        # Get S&P 500 data for returns overlay
        cache = DataCache(settings.db_path)
        spy_df = cache.get_series("SPY")
        if not spy_df.empty:
            # Align with history dates and calculate cumulative return
            start_date = history.index[0]
            spy_aligned = spy_df[spy_df.index >= start_date]
            if not spy_aligned.empty:
                base_price = spy_aligned["value"].iloc[0]
                spy_aligned = spy_aligned.copy()
                spy_aligned["return"] = ((spy_aligned["value"] / base_price) - 1) * 100
                
                # Match to chart dates
                for d in history.index:
                    if d in spy_aligned.index:
                        sp500_returns.append(round(spy_aligned.loc[d, "return"], 2))
                    elif len(sp500_returns) > 0:
                        sp500_returns.append(sp500_returns[-1])  # Forward fill
                    else:
                        sp500_returns.append(0)
    
    # Get regime
    regime = get_regime(result.composite_score)
    
    # Build signal breakdown rows
    sorted_indicators = sorted(
        [(k, v, WEIGHTS.get(k, 0)) for k, v in result.indicators.items()],
        key=lambda x: x[1].percentile,
        reverse=True,
    )
    
    signal_rows = []
    for key, ind, weight in sorted_indicators:
        pct = ind.percentile
        if pct >= 80:
            bar_color, text_color = "#ef4444", "#fca5a5"
        elif pct >= 60:
            bar_color, text_color = "#f97316", "#fdba74"
        elif pct >= 40:
            bar_color, text_color = "#f59e0b", "#fcd34d"
        else:
            bar_color, text_color = "#10b981", "#6ee7b7"
        
        alert = "!" if pct >= 80 else ""
        
        signal_rows.append(f'''
            <div class="signal-row">
                <div class="signal-header">
                    <span class="signal-name">{alert} {ind.name} <span class="signal-weight">{weight*100:.0f}% wt</span></span>
                    <span class="signal-value" style="color: {text_color}">{pct:.0f}</span>
                </div>
                <div class="signal-bar-bg">
                    <div class="signal-bar" style="width: {pct}%; background: {bar_color};"></div>
                </div>
            </div>
        ''')
    
    # Build current levels rows
    level_rows = []
    for key, ind in result.indicators.items():
        formatted = format_value(ind.name, ind.current_value)
        level_rows.append(f'''
            <div class="level-row">
                <span class="level-name">{ind.name}</span>
                <span class="level-value">{formatted}</span>
            </div>
        ''')
    
    # Build alerts
    alerts = generate_alerts(result, history)
    if not alerts:
        alert_html = '''<div class="alert alert-ok">No active alerts</div>'''
    else:
        alert_html = ""
        for level, msg in alerts:
            alert_class = "alert-high" if level == "HIGH" else "alert-elevated"
            alert_html += f'''<div class="alert {alert_class}"><span class="alert-level">{level}:</span> {msg}</div>'''
    
    # Build tab content
    indicators_tab = build_indicators_tab(result)
    backtest_tab = build_backtest_tab()
    models_tab = build_models_tab()
    
    # Market events as JSON for JavaScript
    market_events_json = json.dumps(MARKET_EVENTS)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fast Market Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 1.5rem;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1rem;
            border-bottom: 1px solid #334155;
            margin-bottom: 1rem;
        }}
        .header-title {{ font-size: 1.5rem; font-weight: 600; color: #f1f5f9; }}
        .header-subtitle {{ color: #64748b; font-size: 0.75rem; margin-top: 0.25rem; }}
        .header-meta {{ color: #64748b; font-size: 0.7rem; text-align: right; }}
        
        /* Period Selector */
        .period-selector {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .period-selector label {{ color: #94a3b8; font-size: 0.8rem; }}
        .period-select {{
            background: #1e293b;
            border: 1px solid #334155;
            color: #e2e8f0;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
        }}
        .period-select:focus {{ outline: none; border-color: #3b82f6; }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 2rem;
            border-bottom: 1px solid #334155;
            margin-bottom: 1.5rem;
        }}
        .tab {{
            padding: 0.75rem 0;
            color: #94a3b8;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }}
        .tab:hover {{ color: #e2e8f0; }}
        .tab.active {{ color: #3b82f6; border-bottom-color: #3b82f6; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        /* Regime Header */
        .regime-header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-left: 4px solid {regime['color']};
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .regime-label-small {{ color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }}
        .regime-score-row {{ display: flex; align-items: baseline; gap: 1rem; margin-top: 0.25rem; }}
        .regime-score {{ font-size: 3.5rem; font-weight: 700; color: {regime['color']}; font-family: 'SF Mono', 'Consolas', monospace; line-height: 1; }}
        .regime-delta {{ font-size: 1.25rem; color: {delta_color}; font-family: 'SF Mono', 'Consolas', monospace; }}
        .regime-badge {{ background: {regime['color']}22; border: 1px solid {regime['color']}; color: {regime['color']}; padding: 0.5rem 1.5rem; border-radius: 4px; font-weight: 600; font-size: 1.1rem; }}
        .regime-action {{ color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem; text-align: right; }}
        .regime-footer {{ color: #64748b; font-size: 0.7rem; margin-bottom: 1rem; }}
        
        /* Alerts */
        .alerts-container {{ margin-bottom: 1rem; }}
        .alert {{ border-radius: 4px; padding: 0.5rem 1rem; font-size: 0.8rem; margin-bottom: 0.5rem; }}
        .alert-ok {{ background: #10b98122; border: 1px solid #10b981; color: #6ee7b7; }}
        .alert-high {{ background: #ef444422; border: 1px solid #ef4444; color: #fca5a5; }}
        .alert-elevated {{ background: #f9731622; border: 1px solid #f97316; color: #fdba74; }}
        .alert-level {{ font-weight: 600; }}
        
        /* Chart */
        .chart-container {{ margin-bottom: 1rem; }}
        #chart {{ height: 380px; }}
        
        /* Event Legend */
        .event-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            padding: 0.5rem 0;
            border-top: 1px solid #334155;
            margin-bottom: 1.5rem;
        }}
        .event-item {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}
        .event-bar {{
            width: 3px;
            height: 14px;
            border-radius: 1px;
        }}
        .event-date {{ color: #94a3b8; font-size: 0.75rem; }}
        .event-label {{ color: #e2e8f0; font-size: 0.75rem; }}
        
        /* Bottom Grid */
        .bottom-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
        @media (max-width: 1024px) {{ .bottom-grid {{ grid-template-columns: 1fr; }} }}
        
        /* Cards */
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem 1.5rem; }}
        .card-title {{ color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem; }}
        
        /* Signal Breakdown */
        .signal-row {{ margin-bottom: 0.75rem; }}
        .signal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }}
        .signal-name {{ color: #e2e8f0; font-size: 0.85rem; }}
        .signal-weight {{ color: #64748b; font-size: 0.7rem; margin-left: 0.5rem; }}
        .signal-value {{ font-family: 'SF Mono', monospace; font-size: 0.9rem; font-weight: 600; }}
        .signal-bar-bg {{ background: #0f172a; border-radius: 2px; height: 6px; overflow: hidden; }}
        .signal-bar {{ height: 100%; border-radius: 2px; }}
        
        /* Current Levels */
        .level-row {{ display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #334155; }}
        .level-row:last-child {{ border-bottom: none; }}
        .level-name {{ color: #94a3b8; font-size: 0.8rem; }}
        .level-value {{ color: #e2e8f0; font-family: 'SF Mono', monospace; font-size: 0.85rem; }}
        
        /* Indicators Tab */
        .tab-heading {{ color: #f1f5f9; font-size: 1.5rem; margin-bottom: 0.5rem; }}
        .tab-intro {{ color: #94a3b8; margin-bottom: 1.5rem; }}
        .indicator-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }}
        .indicator-card-header {{ display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem; }}
        .indicator-card-title {{ color: #f1f5f9; font-size: 1.1rem; margin: 0; }}
        .indicator-card-source {{ color: #64748b; font-size: 0.75rem; margin-top: 0.25rem; }}
        .indicator-card-status-wrap {{ text-align: right; }}
        .indicator-status {{ padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem; border: 1px solid; display: inline-block; }}
        .indicator-weight {{ color: #64748b; font-size: 0.7rem; margin-top: 0.25rem; }}
        .indicator-description {{ color: #cbd5e1; font-size: 0.9rem; margin-bottom: 1rem; }}
        .indicator-details {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
        .indicator-detail-box {{ background: #0f172a; padding: 0.75rem; border-radius: 4px; }}
        .indicator-detail-label {{ color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem; }}
        .indicator-detail-value {{ color: #e2e8f0; font-size: 0.85rem; }}
        @media (max-width: 768px) {{ .indicator-details {{ grid-template-columns: 1fr; }} }}
        
        /* Backtest & Models Tabs */
        .section-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem; }}
        .section-title {{ color: #f1f5f9; font-size: 1.1rem; margin: 1.5rem 0 1rem 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
        .metric-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; text-align: center; }}
        .metric-value {{ font-size: 1.5rem; font-weight: 600; margin: 0.25rem 0; }}
        .metric-label {{ color: #64748b; font-size: 0.7rem; text-transform: uppercase; }}
        .metric-sublabel {{ color: #94a3b8; font-size: 0.75rem; }}
        .rankings-table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 1.5rem; }}
        .rankings-table th {{ text-align: left; padding: 0.75rem 1rem; color: #94a3b8; font-size: 0.8rem; border-bottom: 1px solid #334155; }}
        .rankings-table td {{ padding: 0.75rem 1rem; color: #e2e8f0; font-size: 0.85rem; border-bottom: 1px solid #334155; }}
        .rankings-table tr:last-child td {{ border-bottom: none; }}
        .stress-events {{ color: #94a3b8; margin-left: 1.25rem; margin-top: 0.5rem; }}
        .stress-events li {{ margin-bottom: 0.25rem; }}
        .failed {{ color: #ef4444 !important; }}
        .warning {{ color: #f59e0b !important; }}
        
        /* Stage boxes */
        .stage-row {{ display: flex; align-items: flex-start; gap: 0.75rem; margin-bottom: 1rem; }}
        .stage-badge {{ padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; color: white; white-space: nowrap; }}
        .stage-text {{ color: #94a3b8; font-size: 0.9rem; }}
        .stage-text strong {{ color: #e2e8f0; }}
        
        /* Login overlay */
        .login-overlay {{ position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: #0f172a; display: flex; align-items: center; justify-content: center; z-index: 1000; }}
        .login-overlay.hidden {{ display: none; }}
        .login-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 32px; width: 100%; max-width: 360px; text-align: center; }}
        .login-title {{ font-size: 18px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; }}
        .login-subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 24px; }}
        .login-input {{ width: 100%; padding: 12px 16px; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #e2e8f0; font-size: 14px; margin-bottom: 16px; outline: none; }}
        .login-input:focus {{ border-color: #3b82f6; }}
        .login-button {{ width: 100%; padding: 12px 16px; background: #3b82f6; border: none; border-radius: 8px; color: #fff; font-size: 14px; font-weight: 500; cursor: pointer; }}
        .login-button:hover {{ background: #2563eb; }}
        .login-error {{ color: #ef4444; font-size: 13px; margin-top: 12px; display: none; }}
        .dashboard-content {{ display: none; }}
        .dashboard-content.visible {{ display: block; }}
    </style>
</head>
<body>
    {"" if not password_hash else '''
    <div id="loginOverlay" class="login-overlay">
        <div class="login-box">
            <div class="login-title">Fast Market Dashboard</div>
            <div class="login-subtitle">Enter password to continue</div>
            <input type="password" id="passwordInput" class="login-input" placeholder="Password" autofocus>
            <button id="loginButton" class="login-button">Access Dashboard</button>
            <div id="loginError" class="login-error">Incorrect password</div>
        </div>
    </div>
    '''}
    
    <div class="{"dashboard-content" if password_hash else "dashboard-content visible"}" id="dashboardContent">
    <div class="container">
        <div class="header">
            <div>
                <div class="header-title">Market Stress Monitor</div>
                <div class="header-subtitle">Asset Management Risk Dashboard</div>
            </div>
            <div class="header-meta">Data: FRED + Yahoo Finance<br>Refresh: Daily EOD</div>
        </div>
        
        <div class="period-selector">
            <label>History Period:</label>
            <select class="period-select" id="periodSelect">
                <option value="90">90 Days</option>
                <option value="180">180 Days</option>
                <option value="365" selected>1 Year</option>
            </select>
        </div>
        
        <div class="tabs">
            <div class="tab active" data-tab="dashboard">Dashboard</div>
            <div class="tab" data-tab="indicators">Indicators</div>
            <div class="tab" data-tab="backtest">Backtest</div>
            <div class="tab" data-tab="models">Models</div>
        </div>
        
        <!-- Dashboard Tab -->
        <div class="tab-content active" id="tab-dashboard">
            <div class="regime-header">
                <div>
                    <div class="regime-label-small">Market Stress Level</div>
                    <div class="regime-score-row">
                        <span class="regime-score">{result.composite_score:.1f}</span>
                        <span class="regime-delta">{delta_str} (5d)</span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="regime-badge">{regime['label']}</div>
                    <div class="regime-action">{regime['action']}</div>
                </div>
            </div>
            <div class="regime-footer">As of {result.as_of_date.strftime('%Y-%m-%d')} | Updated {datetime.now().strftime('%H:%M')}</div>
            
            <div class="alerts-container">
                {alert_html}
            </div>
            
            <div class="chart-container">
                <div id="chart"></div>
            </div>
            
            <div class="event-legend" id="eventLegend"></div>
            
            <div class="bottom-grid">
                <div class="card">
                    <div class="card-title">Signal Breakdown (by contribution)</div>
                    {"".join(signal_rows)}
                </div>
                <div class="card">
                    <div class="card-title">Current Levels</div>
                    {"".join(level_rows)}
                </div>
            </div>
        </div>
        
        <!-- Indicators Tab -->
        <div class="tab-content" id="tab-indicators">
            {indicators_tab}
        </div>
        
        <!-- Backtest Tab -->
        <div class="tab-content" id="tab-backtest">
            {backtest_tab}
        </div>
        
        <!-- Models Tab -->
        <div class="tab-content" id="tab-models">
            {models_tab}
        </div>
    </div>
    </div>
    
    <script>
        // Market Events
        const MARKET_EVENTS = {market_events_json};
        
        // Chart data
        const CHART_DATA = {{
            dates: {json.dumps(chart_dates)},
            values: {json.dumps(chart_values)},
            sp500: {json.dumps(sp500_returns)}
        }};
        
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.addEventListener('click', () => {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                tab.classList.add('active');
                document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
            }});
        }});
        
        // Period selector
        document.getElementById('periodSelect').addEventListener('change', (e) => {{
            renderChart(parseInt(e.target.value));
        }});
        
        function renderChart(days) {{
            const endIdx = CHART_DATA.dates.length;
            const startIdx = Math.max(0, endIdx - days);
            const dates = CHART_DATA.dates.slice(startIdx);
            const values = CHART_DATA.values.slice(startIdx);
            const sp500 = CHART_DATA.sp500.slice(startIdx);
            
            // Recalculate S&P returns from period start
            const sp500Base = sp500[0] || 0;
            const sp500Adjusted = sp500.map(v => v - sp500Base);
            
            // Filter events for this period
            const startDate = dates[0];
            const endDate = dates[dates.length - 1];
            const visibleEvents = MARKET_EVENTS.filter(e => e.date >= startDate && e.date <= endDate);
            
            // Build shapes for regime bands and event lines
            const shapes = [
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 30, fillcolor: '#10b981', opacity: 0.08, line: {{ width: 0 }}, yref: 'y' }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 50, fillcolor: '#f59e0b', opacity: 0.08, line: {{ width: 0 }}, yref: 'y' }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 50, y1: 70, fillcolor: '#f97316', opacity: 0.08, line: {{ width: 0 }}, yref: 'y' }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 100, fillcolor: '#ef4444', opacity: 0.08, line: {{ width: 0 }}, yref: 'y' }},
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: {{ color: '#475569', width: 1, dash: 'dot' }}, yref: 'y' }},
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 50, y1: 50, line: {{ color: '#475569', width: 1, dash: 'dot' }}, yref: 'y' }},
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 70, line: {{ color: '#475569', width: 1, dash: 'dot' }}, yref: 'y' }},
            ];
            
            visibleEvents.forEach(e => {{
                shapes.push({{
                    type: 'line', x0: e.date, x1: e.date, y0: 0, y1: 1, yref: 'paper',
                    line: {{ color: e.color, width: 1.5, dash: 'dash' }}
                }});
            }});
            
            // Stress score trace
            const traceStress = {{
                x: dates, y: values,
                type: 'scatter', mode: 'lines',
                line: {{ color: '#3b82f6', width: 2 }},
                fill: 'tozeroy', fillcolor: 'rgba(59, 130, 246, 0.1)',
                name: 'Stress Score',
                yaxis: 'y',
                hovertemplate: 'Stress: %{{y:.1f}}<extra></extra>',
            }};
            
            // S&P 500 return trace (secondary axis, inverted)
            const traceSP500 = {{
                x: dates, y: sp500Adjusted,
                type: 'scatter', mode: 'lines',
                line: {{ color: '#a855f7', width: 1.5, dash: 'dot' }},
                name: 'S&P 500 Return',
                yaxis: 'y2',
                hovertemplate: 'S&P: %{{y:+.1f}}%<extra></extra>',
            }};
            
            // Calculate y2 range (inverted so drops appear as rises)
            const minSP = Math.min(...sp500Adjusted);
            const maxSP = Math.max(...sp500Adjusted);
            const spPadding = Math.max(5, (maxSP - minSP) * 0.1);
            
            const layout = {{
                paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                margin: {{ t: 10, r: 60, b: 40, l: 50 }},
                xaxis: {{ showgrid: true, gridcolor: '#1e293b', color: '#64748b', tickfont: {{ size: 10 }}, tickformat: '%b %d' }},
                yaxis: {{ 
                    showgrid: true, gridcolor: '#1e293b', color: '#3b82f6', tickfont: {{ size: 10, color: '#3b82f6' }}, 
                    range: [0, 100], dtick: 25, title: {{ text: 'Stress Score', font: {{ size: 11, color: '#3b82f6' }} }},
                    side: 'left'
                }},
                yaxis2: {{
                    overlaying: 'y', side: 'right',
                    showgrid: false, color: '#a855f7', tickfont: {{ size: 10, color: '#a855f7' }},
                    range: [maxSP + spPadding, minSP - spPadding],  // Inverted
                    ticksuffix: '%',
                    title: {{ text: 'S&P 500 Return', font: {{ size: 11, color: '#a855f7' }} }}
                }},
                shapes: shapes,
                hovermode: 'x unified',
                showlegend: true,
                legend: {{ orientation: 'h', x: 0.5, xanchor: 'center', y: 1.02, font: {{ size: 10, color: '#94a3b8' }} }},
            }};
            
            Plotly.newPlot('chart', [traceStress, traceSP500], layout, {{ displayModeBar: false, responsive: true }});
            
            // Update event legend
            const legendEl = document.getElementById('eventLegend');
            if (visibleEvents.length > 0) {{
                legendEl.innerHTML = visibleEvents.map(e => {{
                    const dateStr = new Date(e.date).toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }});
                    return '<div class="event-item"><div class="event-bar" style="background: ' + e.color + ';"></div><span class="event-date">' + dateStr + '</span><span class="event-label">' + e.label + '</span></div>';
                }}).join('');
            }} else {{
                legendEl.innerHTML = '';
            }}
        }}
        
        // Initialize chart
        renderChart(365);
    </script>
    {"" if not password_hash else f'''
    <script>
        const HASH = "{password_hash}";
        async function sha256(text) {{
            const encoder = new TextEncoder();
            const data = encoder.encode(text);
            const hashBuffer = await crypto.subtle.digest("SHA-256", data);
            const hashArray = Array.from(new Uint8Array(hashBuffer));
            return hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
        }}
        async function checkAuth() {{
            const stored = sessionStorage.getItem("dashboard_auth");
            if (stored === HASH) showDashboard();
        }}
        function showDashboard() {{
            document.getElementById("loginOverlay").classList.add("hidden");
            document.getElementById("dashboardContent").classList.add("visible");
        }}
        async function handleLogin() {{
            const password = document.getElementById("passwordInput").value;
            const hash = await sha256(password);
            if (hash === HASH) {{
                sessionStorage.setItem("dashboard_auth", hash);
                showDashboard();
            }} else {{
                document.getElementById("loginError").style.display = "block";
                document.getElementById("passwordInput").value = "";
            }}
        }}
        document.getElementById("loginButton").addEventListener("click", handleLogin);
        document.getElementById("passwordInput").addEventListener("keypress", (e) => {{
            if (e.key === "Enter") handleLogin();
        }});
        checkAuth();
    </script>
    '''}
</body>
</html>'''
    
    # Determine output path
    if output_path is None:
        output_path = settings.cache_dir.parent / "dist" / "index.html"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    
    return output_path


def main() -> None:
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export dashboard as HTML")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path (default: dist/index.html)")
    parser.add_argument("-p", "--password", type=str, default=None, help="Password for client-side protection")
    args = parser.parse_args()
    
    try:
        path = export_html(args.output, args.password)
        print(f"Dashboard exported to: {path}")
        print(f"File size: {path.stat().st_size / 1024:.1f} KB")
        if args.password or os.getenv("DASHBOARD_PASSWORD"):
            print("Password protection: enabled")
    except ValueError as e:
        print(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
