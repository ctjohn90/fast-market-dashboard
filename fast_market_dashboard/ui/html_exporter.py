"""Export dashboard as self-contained HTML."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

from fast_market_dashboard.config import Settings
from fast_market_dashboard.indicators.calculator import IndicatorCalculator


def hash_password(password: str) -> str:
    """Create SHA-256 hash of password."""
    return hashlib.sha256(password.encode()).hexdigest()


def get_stress_color(score: float) -> str:
    """Return color based on stress level."""
    if score < 30:
        return "#22c55e"  # green
    elif score < 50:
        return "#eab308"  # yellow
    elif score < 70:
        return "#f97316"  # orange
    else:
        return "#ef4444"  # red


def get_stress_bg(score: float) -> str:
    """Return background color based on stress level."""
    if score < 30:
        return "rgba(34, 197, 94, 0.1)"
    elif score < 50:
        return "rgba(234, 179, 8, 0.1)"
    elif score < 70:
        return "rgba(249, 115, 22, 0.1)"
    else:
        return "rgba(239, 68, 68, 0.1)"


def generate_sparkline_svg(values: list[float], color: str, width: int = 120, height: int = 32) -> str:
    """Generate inline SVG sparkline."""
    if not values or len(values) < 2:
        return ""
    
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val != min_val else 1
    
    # Normalize to SVG coordinates
    points = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * width
        y = height - ((v - min_val) / val_range * (height - 4) + 2)
        points.append(f"{x:.1f},{y:.1f}")
    
    path = "M" + " L".join(points)
    
    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        <path d="{path}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>'''


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
    
    # Prepare chart data
    chart_dates = []
    chart_values = []
    if not history.empty:
        chart_dates = [d.strftime("%Y-%m-%d") for d in history.index]
        chart_values = [round(v, 1) for v in history["composite"].values]
    
    # Build indicator rows
    indicator_rows = []
    for key, ind in result.indicators.items():
        color = get_stress_color(ind.percentile)
        sparkline_values = ind.history.values.tolist() if hasattr(ind.history, 'values') else list(ind.history)
        sparkline = generate_sparkline_svg(sparkline_values, color)
        
        indicator_rows.append(f'''
            <tr>
                <td class="indicator-name">{ind.name}</td>
                <td class="indicator-value">{ind.current_value:.2f}</td>
                <td class="indicator-percentile" style="color: {color}">{ind.percentile:.0f}</td>
                <td class="indicator-sparkline">{sparkline}</td>
            </tr>
        ''')
    
    # Stress level styling
    composite_color = get_stress_color(result.composite_score)
    composite_bg = get_stress_bg(result.composite_score)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fast Market Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0a0a;
            color: #e5e5e5;
            min-height: 100vh;
            padding: 24px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #262626;
        }}
        
        h1 {{
            font-size: 20px;
            font-weight: 600;
            color: #fafafa;
        }}
        
        .meta {{
            font-size: 13px;
            color: #737373;
        }}
        
        .score-card {{
            background: {composite_bg};
            border: 1px solid #262626;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            text-align: center;
        }}
        
        .score-value {{
            font-size: 72px;
            font-weight: 700;
            color: {composite_color};
            line-height: 1;
            margin-bottom: 8px;
        }}
        
        .score-label {{
            font-size: 14px;
            color: #a3a3a3;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }}
        
        .stress-level {{
            display: inline-block;
            padding: 6px 16px;
            background: {composite_color}22;
            color: {composite_color};
            border-radius: 6px;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        @media (max-width: 768px) {{
            .grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .card {{
            background: #141414;
            border: 1px solid #262626;
            border-radius: 12px;
            padding: 20px;
        }}
        
        .card-title {{
            font-size: 13px;
            color: #737373;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th {{
            text-align: left;
            font-size: 11px;
            color: #525252;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 8px 0;
            border-bottom: 1px solid #262626;
        }}
        
        td {{
            padding: 12px 0;
            border-bottom: 1px solid #1a1a1a;
            vertical-align: middle;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .indicator-name {{
            font-size: 14px;
            color: #d4d4d4;
        }}
        
        .indicator-value {{
            font-size: 14px;
            color: #737373;
            text-align: right;
            padding-right: 16px;
        }}
        
        .indicator-percentile {{
            font-size: 16px;
            font-weight: 600;
            text-align: right;
            padding-right: 16px;
        }}
        
        .indicator-sparkline {{
            text-align: right;
        }}
        
        #chart {{
            height: 300px;
        }}
        
        .chart-card {{
            grid-column: 1 / -1;
        }}
        
        .thresholds {{
            display: flex;
            gap: 24px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #262626;
        }}
        
        .threshold {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #737373;
        }}
        
        .threshold-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        
        /* Login overlay styles */
        .login-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: #0a0a0a;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }}
        
        .login-overlay.hidden {{
            display: none;
        }}
        
        .login-box {{
            background: #141414;
            border: 1px solid #262626;
            border-radius: 12px;
            padding: 32px;
            width: 100%;
            max-width: 360px;
            text-align: center;
        }}
        
        .login-title {{
            font-size: 18px;
            font-weight: 600;
            color: #fafafa;
            margin-bottom: 8px;
        }}
        
        .login-subtitle {{
            font-size: 13px;
            color: #737373;
            margin-bottom: 24px;
        }}
        
        .login-input {{
            width: 100%;
            padding: 12px 16px;
            background: #0a0a0a;
            border: 1px solid #262626;
            border-radius: 8px;
            color: #e5e5e5;
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
        <header>
            <h1>Fast Market Dashboard</h1>
            <div class="meta">
                As of {result.as_of_date.strftime("%B %d, %Y")} | Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </header>
        
        <div class="score-card">
            <div class="score-value">{result.composite_score:.0f}</div>
            <div class="score-label">Composite Stress Score</div>
            <div class="stress-level">{result.stress_level}</div>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">Component Breakdown</div>
                <table>
                    <thead>
                        <tr>
                            <th>Indicator</th>
                            <th style="text-align: right; padding-right: 16px;">Value</th>
                            <th style="text-align: right; padding-right: 16px;">%ile</th>
                            <th style="text-align: right;">30d</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(indicator_rows)}
                    </tbody>
                </table>
            </div>
            
            <div class="card">
                <div class="card-title">Score Interpretation</div>
                <div style="margin-top: 8px;">
                    <div class="threshold">
                        <div class="threshold-dot" style="background: #22c55e;"></div>
                        <span>0-30: Calm markets</span>
                    </div>
                    <div class="threshold" style="margin-top: 12px;">
                        <div class="threshold-dot" style="background: #eab308;"></div>
                        <span>30-50: Normal conditions</span>
                    </div>
                    <div class="threshold" style="margin-top: 12px;">
                        <div class="threshold-dot" style="background: #f97316;"></div>
                        <span>50-70: Elevated stress</span>
                    </div>
                    <div class="threshold" style="margin-top: 12px;">
                        <div class="threshold-dot" style="background: #ef4444;"></div>
                        <span>70-100: Fast market / high stress</span>
                    </div>
                </div>
                <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid #262626;">
                    <div style="font-size: 12px; color: #525252; margin-bottom: 8px;">DATA SOURCES</div>
                    <div style="font-size: 13px; color: #737373;">
                        FRED (Federal Reserve), Yahoo Finance
                    </div>
                </div>
            </div>
            
            <div class="card chart-card">
                <div class="card-title">90-Day Composite History</div>
                <div id="chart"></div>
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
            line: {{
                color: '#3b82f6',
                width: 2
            }},
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)'
        }};
        
        const layout = {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{ t: 10, r: 20, b: 40, l: 40 }},
            xaxis: {{
                showgrid: false,
                color: '#525252',
                tickfont: {{ size: 11 }}
            }},
            yaxis: {{
                showgrid: true,
                gridcolor: '#1a1a1a',
                color: '#525252',
                tickfont: {{ size: 11 }},
                range: [0, 100]
            }},
            shapes: [
                {{
                    type: 'line',
                    x0: dates[0],
                    x1: dates[dates.length - 1],
                    y0: 30,
                    y1: 30,
                    line: {{ color: '#22c55e', width: 1, dash: 'dot' }}
                }},
                {{
                    type: 'line',
                    x0: dates[0],
                    x1: dates[dates.length - 1],
                    y0: 50,
                    y1: 50,
                    line: {{ color: '#eab308', width: 1, dash: 'dot' }}
                }},
                {{
                    type: 'line',
                    x0: dates[0],
                    x1: dates[dates.length - 1],
                    y0: 70,
                    y1: 70,
                    line: {{ color: '#ef4444', width: 1, dash: 'dot' }}
                }}
            ]
        }};
        
        const config = {{
            displayModeBar: false,
            responsive: true
        }};
        
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
