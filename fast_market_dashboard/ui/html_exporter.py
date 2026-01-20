"""Export dashboard as self-contained HTML."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from fast_market_dashboard.config import Settings
from fast_market_dashboard.indicators.calculator import IndicatorCalculator, WEIGHTS


def hash_password(password: str) -> str:
    """Create SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_regime(score: float) -> dict:
    """Get regime info for a given score."""
    if score < 30:
        return {"color": "#10b981", "label": "CALM", "action": "Risk-on conditions"}
    elif score < 50:
        return {"color": "#f59e0b", "label": "NORMAL", "action": "Standard monitoring"}
    elif score < 70:
        return {"color": "#f97316", "label": "ELEVATED", "action": "Increased vigilance"}
    else:
        return {"color": "#ef4444", "label": "FAST MARKET", "action": "Active risk management"}


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


def export_html(output_path: Path | str | None = None, password: str | None = None) -> Path:
    """
    Generate self-contained HTML dashboard.
    
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
    
    # Get history for chart
    history = calculator.get_history(days=90)
    
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
    
    # Prepare chart data
    chart_dates = []
    chart_values = []
    if not history.empty:
        chart_dates = [d.strftime("%Y-%m-%d") for d in history.index]
        chart_values = [round(v, 1) for v in history["composite"].values]
    
    # Get regime
    regime = get_regime(result.composite_score)
    
    # Build signal breakdown rows (sorted by contribution)
    sorted_indicators = sorted(
        [(k, v, WEIGHTS.get(k, 0)) for k, v in result.indicators.items()],
        key=lambda x: x[1].percentile * x[2],
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
            if level == "HIGH":
                alert_class = "alert-high"
            else:
                alert_class = "alert-elevated"
            alert_html += f'''<div class="alert {alert_class}"><span class="alert-level">{level}:</span> {msg}</div>'''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fast Market Dashboard Testing</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
            padding: 1.5rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1rem;
            border-bottom: 1px solid #334155;
            margin-bottom: 1rem;
        }}
        
        .header-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #f1f5f9;
        }}
        
        .header-subtitle {{
            color: #64748b;
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }}
        
        .header-meta {{
            color: #64748b;
            font-size: 0.7rem;
            text-align: right;
        }}
        
        /* Regime Header */
        .regime-header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-left: 4px solid {regime['color']};
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .regime-label-small {{
            color: #94a3b8;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }}
        
        .regime-score-row {{
            display: flex;
            align-items: baseline;
            gap: 1rem;
            margin-top: 0.25rem;
        }}
        
        .regime-score {{
            font-size: 3.5rem;
            font-weight: 700;
            color: {regime['color']};
            font-family: 'SF Mono', 'Consolas', monospace;
            line-height: 1;
        }}
        
        .regime-delta {{
            font-size: 1.25rem;
            color: {delta_color};
            font-family: 'SF Mono', 'Consolas', monospace;
        }}
        
        .regime-badge {{
            background: {regime['color']}22;
            border: 1px solid {regime['color']};
            color: {regime['color']};
            padding: 0.5rem 1.5rem;
            border-radius: 4px;
            font-weight: 600;
            font-size: 1.1rem;
        }}
        
        .regime-action {{
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            text-align: right;
        }}
        
        .regime-footer {{
            color: #64748b;
            font-size: 0.7rem;
            margin-top: 1rem;
            border-top: 1px solid #334155;
            padding-top: 0.75rem;
        }}
        
        /* Grid Layout */
        .main-grid {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .bottom-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}
        
        @media (max-width: 1024px) {{
            .main-grid, .bottom-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Cards */
        .card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem 1.5rem;
        }}
        
        .card-title {{
            color: #94a3b8;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
        }}
        
        /* Chart */
        #chart {{
            height: 280px;
        }}
        
        /* Alerts */
        .alert {{
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }}
        
        .alert-ok {{
            background: #10b98122;
            border: 1px solid #10b981;
            color: #6ee7b7;
        }}
        
        .alert-high {{
            background: #ef444422;
            border: 1px solid #ef4444;
            color: #fca5a5;
        }}
        
        .alert-elevated {{
            background: #f9731622;
            border: 1px solid #f97316;
            color: #fdba74;
        }}
        
        .alert-level {{
            font-weight: 600;
        }}
        
        /* Signal Breakdown */
        .signal-row {{
            margin-bottom: 0.75rem;
        }}
        
        .signal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.25rem;
        }}
        
        .signal-name {{
            color: #e2e8f0;
            font-size: 0.85rem;
        }}
        
        .signal-weight {{
            color: #64748b;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }}
        
        .signal-value {{
            font-family: 'SF Mono', monospace;
            font-size: 0.9rem;
            font-weight: 600;
        }}
        
        .signal-bar-bg {{
            background: #0f172a;
            border-radius: 2px;
            height: 6px;
            overflow: hidden;
        }}
        
        .signal-bar {{
            height: 100%;
            border-radius: 2px;
        }}
        
        /* Current Levels */
        .level-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.4rem 0;
            border-bottom: 1px solid #334155;
        }}
        
        .level-row:last-child {{
            border-bottom: none;
        }}
        
        .level-name {{
            color: #94a3b8;
            font-size: 0.8rem;
        }}
        
        .level-value {{
            color: #e2e8f0;
            font-family: 'SF Mono', monospace;
            font-size: 0.85rem;
        }}
        
        /* Login overlay styles */
        .login-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: #0f172a;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}
        
        .login-overlay.hidden {{
            display: none;
        }}
        
        .login-box {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 32px;
            width: 100%;
            max-width: 360px;
            text-align: center;
        }}
        
        .login-title {{
            font-size: 18px;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 8px;
        }}
        
        .login-subtitle {{
            font-size: 13px;
            color: #64748b;
            margin-bottom: 24px;
        }}
        
        .login-input {{
            width: 100%;
            padding: 12px 16px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 14px;
            margin-bottom: 16px;
            outline: none;
        }}
        
        .login-input:focus {{
            border-color: #3b82f6;
        }}
        
        .login-button {{
            width: 100%;
            padding: 12px 16px;
            background: #3b82f6;
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
        }}
        
        .login-button:hover {{
            background: #2563eb;
        }}
        
        .login-error {{
            color: #ef4444;
            font-size: 13px;
            margin-top: 12px;
            display: none;
        }}
        
        .dashboard-content {{
            display: none;
        }}
        
        .dashboard-content.visible {{
            display: block;
        }}
    </style>
</head>
<body>
    {"" if not password_hash else '''
    <div id="loginOverlay" class="login-overlay">
        <div class="login-box">
            <div class="login-title">Fast Market Dashboard Testing</div>
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
            <div class="header-meta">
                Data: FRED + Yahoo Finance<br>Refresh: Daily EOD
            </div>
        </div>
        
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
        <div class="regime-footer">
            As of {result.as_of_date.strftime('%Y-%m-%d')} | Updated {datetime.now().strftime('%H:%M')}
        </div>
        
        <div class="main-grid">
            <div class="card">
                <div class="card-title">90-Day Composite History</div>
                <div id="chart"></div>
            </div>
            
            <div class="card">
                <div class="card-title">Alerts</div>
                {alert_html}
            </div>
        </div>
        
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
    </div>
    
    <script>
        const dates = {json.dumps(chart_dates)};
        const values = {json.dumps(chart_values)};
        
        const trace = {{
            x: dates,
            y: values,
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#3b82f6', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)',
            hovertemplate: '%{{x|%b %d, %Y}}<br>Score: %{{y:.1f}}<extra></extra>',
        }};
        
        const layout = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{ t: 10, r: 10, b: 40, l: 40 }},
            xaxis: {{
                showgrid: true,
                gridcolor: '#1e293b',
                color: '#64748b',
                tickfont: {{ size: 10 }},
                tickformat: '%b %d',
            }},
            yaxis: {{
                showgrid: true,
                gridcolor: '#1e293b',
                color: '#64748b',
                tickfont: {{ size: 10 }},
                range: [0, 100],
                dtick: 25,
            }},
            shapes: [
                // Regime bands
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 30, fillcolor: '#10b981', opacity: 0.08, line: {{ width: 0 }} }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 50, fillcolor: '#f59e0b', opacity: 0.08, line: {{ width: 0 }} }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 50, y1: 70, fillcolor: '#f97316', opacity: 0.08, line: {{ width: 0 }} }},
                {{ type: 'rect', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 100, fillcolor: '#ef4444', opacity: 0.08, line: {{ width: 0 }} }},
                // Threshold lines
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 30, y1: 30, line: {{ color: '#475569', width: 1, dash: 'dot' }} }},
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 50, y1: 50, line: {{ color: '#475569', width: 1, dash: 'dot' }} }},
                {{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 70, y1: 70, line: {{ color: '#475569', width: 1, dash: 'dot' }} }},
            ],
            hovermode: 'x unified',
        }};
        
        const config = {{ displayModeBar: false, responsive: true }};
        
        Plotly.newPlot('chart', [trace], layout, config);
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
            if (stored === HASH) {{
                showDashboard();
            }}
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
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path (default: dist/index.html)"
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        default=None,
        help="Password for client-side protection (or set DASHBOARD_PASSWORD env var)"
    )
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
