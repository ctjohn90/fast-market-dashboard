"""Streamlit dashboard for market stress visualization.

Multi-tab dashboard for asset management risk teams:
- Dashboard: Real-time stress monitoring
- Indicators: Documentation for each indicator
- Backtest: Historical performance analysis
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from fast_market_dashboard.indicators import IndicatorCalculator
from fast_market_dashboard.indicators.calculator import CompositeResult, WEIGHTS, INDICATOR_INFO
from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings


# Regime definitions
REGIMES = {
    "calm": {"range": (0, 30), "color": "#10b981", "label": "CALM", "action": "Risk-on conditions"},
    "normal": {"range": (30, 50), "color": "#f59e0b", "label": "NORMAL", "action": "Standard monitoring"},
    "elevated": {"range": (50, 70), "color": "#f97316", "label": "ELEVATED", "action": "Increased vigilance"},
    "fast_market": {"range": (70, 100), "color": "#ef4444", "label": "FAST MARKET", "action": "Active risk management"},
}


def get_regime(score: float) -> dict:
    """Get regime info for a given score."""
    for regime in REGIMES.values():
        if regime["range"][0] <= score < regime["range"][1]:
            return regime
    return REGIMES["fast_market"]


def format_delta(current: float, previous: float) -> tuple[str, str]:
    """Format change with color."""
    delta = current - previous
    if abs(delta) < 0.5:
        return "0", "#6b7280"
    elif delta > 0:
        return f"+{delta:.1f}", "#ef4444"
    else:
        return f"{delta:.1f}", "#10b981"


# =============================================================================
# TAB 1: DASHBOARD
# =============================================================================

def render_regime_header(result: CompositeResult, history: pd.DataFrame) -> None:
    """Render the main regime indicator."""
    regime = get_regime(result.composite_score)
    
    if len(history) >= 5:
        prev_score = history["composite"].iloc[-5]
        delta_str, delta_color = format_delta(result.composite_score, prev_score)
    else:
        delta_str, delta_color = "N/A", "#6b7280"
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-left: 4px solid {regime['color']};
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin-bottom: 1rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em;">
                        Market Stress Level
            </div>
                    <div style="display: flex; align-items: baseline; gap: 1rem; margin-top: 0.25rem;">
                        <span style="font-size: 3.5rem; font-weight: 700; color: {regime['color']}; font-family: 'SF Mono', 'Consolas', monospace;">
                {result.composite_score:.1f}
                        </span>
                        <span style="font-size: 1.25rem; color: {delta_color}; font-family: 'SF Mono', 'Consolas', monospace;">
                            {delta_str} (5d)
                        </span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="
                        background: {regime['color']}22;
                        border: 1px solid {regime['color']};
                        color: {regime['color']};
                        padding: 0.5rem 1.5rem;
                        border-radius: 4px;
                        font-weight: 600;
                        font-size: 1.1rem;
                    ">
                        {regime['label']}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.5rem;">
                        {regime['action']}
                    </div>
            </div>
            </div>
            <div style="color: #64748b; font-size: 0.7rem; margin-top: 1rem; border-top: 1px solid #334155; padding-top: 0.75rem;">
                As of {result.as_of_date.strftime('%Y-%m-%d')} | Updated {datetime.now().strftime('%H:%M')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_indicator_panel(result: CompositeResult) -> None:
    """Render indicator panel with signal strength bars."""
    sorted_indicators = sorted(
        [(k, v, WEIGHTS.get(k, 0)) for k, v in result.indicators.items()],
        key=lambda x: x[1].percentile * x[2],
        reverse=True,
    )
    
    st.markdown(
        """<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem 1.5rem;">
            <div style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">
                Signal Breakdown (by contribution)
            </div>""",
        unsafe_allow_html=True,
    )
    
    for key, indicator, weight in sorted_indicators:
        pct = indicator.percentile
        if pct >= 80:
            bar_color, text_color = "#ef4444", "#fca5a5"
        elif pct >= 60:
            bar_color, text_color = "#f97316", "#fdba74"
        elif pct >= 40:
            bar_color, text_color = "#f59e0b", "#fcd34d"
        else:
            bar_color, text_color = "#10b981", "#6ee7b7"
        
        alert = "!" if pct >= 80 else ""
        
            st.markdown(
            f"""<div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
                    <span style="color: #e2e8f0; font-size: 0.85rem;">
                        {alert} {indicator.name}
                        <span style="color: #64748b; font-size: 0.7rem; margin-left: 0.5rem;">{weight*100:.0f}% wt</span>
                    </span>
                    <span style="color: {text_color}; font-family: 'SF Mono', monospace; font-size: 0.9rem; font-weight: 600;">{pct:.0f}</span>
                </div>
                <div style="background: #0f172a; border-radius: 2px; height: 6px; overflow: hidden;">
                    <div style="background: {bar_color}; width: {pct}%; height: 100%;"></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_chart(history: pd.DataFrame, days: int = 90) -> None:
    """Render composite history with regime bands."""
    if history.empty:
        st.info("Insufficient history for chart")
        return
    
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=30, fillcolor="#10b981", opacity=0.08, line_width=0)
    fig.add_hrect(y0=30, y1=50, fillcolor="#f59e0b", opacity=0.08, line_width=0)
    fig.add_hrect(y0=50, y1=70, fillcolor="#f97316", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.08, line_width=0)
    
    for thresh in [30, 50, 70]:
        fig.add_hline(y=thresh, line_dash="dot", line_color="#475569", line_width=1)
    
                fig.add_trace(go.Scatter(
        x=history.index, y=history["composite"],
        mode="lines", line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59, 130, 246, 0.1)",
        hovertemplate="%{x|%b %d, %Y}<br>Score: %{y:.1f}<extra></extra>",
    ))
    
                fig.add_trace(go.Scatter(
        x=[history.index[-1]], y=[history["composite"].iloc[-1]],
        mode="markers", marker=dict(color=get_regime(history["composite"].iloc[-1])["color"], size=10),
        hoverinfo="skip",
                ))
                
                fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
        title=dict(text=f"{days}-Day Composite History", font=dict(size=12, color="#94a3b8"), x=0),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10), tickformat="%b %d"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", tickfont=dict(color="#64748b", size=10), range=[0, 100], dtick=25),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_raw_values(result: CompositeResult) -> None:
    """Render current raw values for context."""
    st.markdown(
        """<div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem 1.5rem;">
            <div style="color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;">
                Current Levels
            </div>""",
        unsafe_allow_html=True,
    )
    
    for key, indicator in result.indicators.items():
        val = indicator.current_value
        if "Spread" in indicator.name or "Yield" in indicator.name:
            formatted = f"{val:.2f}%"
        elif "VIX" in indicator.name and "Term" not in indicator.name:
            formatted = f"{val:.1f}"
        elif "Term Structure" in indicator.name:
            status = "backwardation" if val < 1 else "contango"
            formatted = f"{val:.3f} ({status})"
        elif "USD" in indicator.name:
            formatted = f"{val:.1f}"
        elif "Drawdown" in indicator.name:
            formatted = f"{val:.0f}"
        elif "Correlation" in indicator.name:
            formatted = f"{val:.3f}"
        elif "/" in indicator.name or "Rotation" in indicator.name:
            formatted = f"{val:.3f}"
            else:
            formatted = f"{val:.2f}"
        
        st.markdown(
            f"""<div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #334155;">
                <span style="color: #94a3b8; font-size: 0.8rem;">{indicator.name}</span>
                <span style="color: #e2e8f0; font-family: 'SF Mono', monospace; font-size: 0.85rem;">{formatted}</span>
            </div>""",
            unsafe_allow_html=True,
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_alerts(result: CompositeResult, history: pd.DataFrame) -> None:
    """Render active alerts based on thresholds."""
    alerts = []
    
    for key, ind in result.indicators.items():
        if ind.percentile >= 90:
            alerts.append(("HIGH", f"{ind.name} at {ind.percentile:.0f}th percentile"))
        elif ind.percentile >= 80:
            alerts.append(("ELEVATED", f"{ind.name} at {ind.percentile:.0f}th percentile"))
    
    if len(history) >= 5:
        change_5d = result.composite_score - history["composite"].iloc[-5]
        if change_5d >= 20:
            alerts.append(("HIGH", f"Composite up {change_5d:.0f} pts in 5 days"))
        elif change_5d >= 15:
            alerts.append(("ELEVATED", f"Composite up {change_5d:.0f} pts in 5 days"))
    
    if result.composite_score >= 70:
        alerts.append(("HIGH", "Fast market conditions active"))
    elif result.composite_score >= 50:
        alerts.append(("ELEVATED", "Elevated stress regime"))
    
    if not alerts:
        st.markdown(
            """<div style="background: #10b98122; border: 1px solid #10b981; border-radius: 4px; padding: 0.75rem 1rem; color: #6ee7b7; font-size: 0.85rem;">
                No active alerts
            </div>""",
            unsafe_allow_html=True,
        )
        return
    
    for level, msg in alerts[:5]:
        if level == "HIGH":
            bg, border, text = "#ef444422", "#ef4444", "#fca5a5"
        else:
            bg, border, text = "#f9731622", "#f97316", "#fdba74"
        st.markdown(
            f"""<div style="background: {bg}; border: 1px solid {border}; border-radius: 4px; padding: 0.5rem 1rem; color: {text}; font-size: 0.8rem; margin-bottom: 0.5rem;">
                <span style="font-weight: 600;">{level}:</span> {msg}
            </div>""",
            unsafe_allow_html=True,
        )


def render_dashboard_tab(calc: IndicatorCalculator, result: CompositeResult, history: pd.DataFrame, days: int = 90) -> None:
    """Render the main dashboard tab."""
    render_regime_header(result, history)
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        render_history_chart(history, days)
    
    with col_right:
        st.markdown("<div style='color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;'>Alerts</div>", unsafe_allow_html=True)
        render_alerts(result, history)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        render_indicator_panel(result)
    with col2:
        render_raw_values(result)


# =============================================================================
# TAB 2: INDICATOR DESCRIPTIONS
# =============================================================================

def render_indicators_tab(result: CompositeResult) -> None:
    """Render the indicator descriptions tab."""
    st.markdown("## Indicator Reference Guide")
    st.markdown("Each indicator in the composite score is selected based on backtesting against historical stress events.")
    
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
        else:
            pct = None
            status_color = "#6b7280"
            status = "No Data"
        
        st.markdown(
            f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                    <div>
                        <h3 style="color: #f1f5f9; margin: 0; font-size: 1.1rem;">{info['name']}</h3>
                        <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">{info['source']}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="background: {status_color}22; border: 1px solid {status_color}; color: {status_color}; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.8rem;">
                            {status}: {f'{pct:.0f}' if pct is not None else 'N/A'}
                        </div>
                        <div style="color: #64748b; font-size: 0.7rem; margin-top: 0.25rem;">Weight: {weight:.0f}%</div>
                    </div>
                </div>
                <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0 0 1rem 0;">{info['description']}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div style="background: #0f172a; padding: 0.75rem; border-radius: 4px;">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Interpretation</div>
                        <div style="color: #e2e8f0; font-size: 0.85rem;">{info['interpretation']}</div>
                    </div>
                    <div style="background: #0f172a; padding: 0.75rem; border-radius: 4px;">
                        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 0.25rem;">Backtest Performance</div>
                        <div style="color: #e2e8f0; font-size: 0.85rem;">Signal Ratio: {info['signal_ratio']}x | Detection: {info['detection_rate']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 3: BACKTESTING RESULTS
# =============================================================================

def render_backtest_tab() -> None:
    """Render the backtesting results tab."""
    st.markdown("## Backtesting Methodology & Results")
    
    st.markdown("""
    ### How We Test Indicators
    
    Each indicator is evaluated against **6 major market stress events**:
    - **2008-09**: Financial Crisis (56.8% drawdown)
    - **2011**: Debt Ceiling Crisis (19.4% drawdown)
    - **2015**: China Devaluation (12.4% drawdown)
    - **2018**: Fed Tightening (19.8% drawdown)
    - **2020**: COVID Crash (33.9% drawdown)
    - **2022**: Rate Shock (25.4% drawdown)
    """)
    
    st.markdown("### Key Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Signal Ratio</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Average percentile during stress / Average percentile during normal times.
                <br><br>
                <strong style="color: #10b981;">Higher is better.</strong> A ratio of 2.0x means the indicator is twice as elevated during stress vs. normal times.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Detection Rate</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Percentage of stress events where the indicator exceeded the 70th percentile.
                <br><br>
                <strong style="color: #10b981;">Higher is better.</strong> 100% means the indicator caught every major stress event.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Indicator Rankings")
    
    # Create ranking table
    rankings = [
        {"Indicator": "HY Credit Spread", "Signal Ratio": "2.19x", "Detection": "100%", "Tier": "1"},
        {"Indicator": "BBB Credit Spread", "Signal Ratio": "2.18x", "Detection": "100%", "Tier": "1"},
        {"Indicator": "VIX", "Signal Ratio": "1.89x", "Detection": "100%", "Tier": "2"},
        {"Indicator": "Defensive Rotation", "Signal Ratio": "1.89x", "Detection": "100%", "Tier": "2"},
        {"Indicator": "USD Index", "Signal Ratio": "1.73x", "Detection": "100%", "Tier": "2"},
        {"Indicator": "Sector Correlation", "Signal Ratio": "1.75x", "Detection": "100%", "Tier": "3"},
        {"Indicator": "VIX Term Structure", "Signal Ratio": "1.63x", "Detection": "100%", "Tier": "3"},
        {"Indicator": "Safe Haven (GLD/SPY)", "Signal Ratio": "1.38x", "Detection": "100%", "Tier": "4"},
        {"Indicator": "S&P 500 Drawdown", "Signal Ratio": "N/A", "Detection": "100%", "Tier": "4"},
    ]
    
    df = pd.DataFrame(rankings)
    
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Indicator": st.column_config.TextColumn("Indicator"),
            "Signal Ratio": st.column_config.TextColumn("Signal Ratio"),
            "Detection": st.column_config.TextColumn("Detection Rate"),
            "Tier": st.column_config.TextColumn("Weight Tier"),
        }
    )
    
    st.markdown("### What Didn't Work")
    
    st.markdown("""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #334155;">
                <th style="text-align: left; padding: 0.5rem; color: #94a3b8;">Indicator</th>
                <th style="text-align: left; padding: 0.5rem; color: #94a3b8;">Signal Ratio</th>
                <th style="text-align: left; padding: 0.5rem; color: #94a3b8;">Problem</th>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 0.5rem; color: #ef4444;">RSI (SPY, QQQ, IWM)</td>
                <td style="padding: 0.5rem; color: #ef4444;">0.53-0.58x</td>
                <td style="padding: 0.5rem; color: #94a3b8;">Goes DOWN during stress (inverted signal)</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 0.5rem; color: #ef4444;">10Y-2Y Spread</td>
                <td style="padding: 0.5rem; color: #ef4444;">0.61x</td>
                <td style="padding: 0.5rem; color: #94a3b8;">Only 33% detection rate - too slow</td>
            </tr>
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 0.5rem; color: #ef4444;">SKEW Index</td>
                <td style="padding: 0.5rem; color: #ef4444;">0.22x</td>
                <td style="padding: 0.5rem; color: #94a3b8;">Opposite of expected behavior</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem; color: #f59e0b;">Fed Funds Rate</td>
                <td style="padding: 0.5rem; color: #f59e0b;">1.02x</td>
                <td style="padding: 0.5rem; color: #94a3b8;">Policy-driven, not market-driven</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Composite Score Weights")
    
    st.markdown("""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
        <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Weighting Rationale</h4>
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0 0 0.75rem 0;">
            Weights are assigned based on three factors:
        </p>
        <ol style="color: #94a3b8; font-size: 0.85rem; margin: 0; padding-left: 1.25rem;">
            <li style="margin-bottom: 0.5rem;">
                <strong style="color: #e2e8f0;">Signal Ratio</strong> - Higher signal ratios (indicator more elevated during stress) get more weight. 
                Credit spreads (2.18-2.19x) outperformed VIX (1.89x) in distinguishing stress from normal.
            </li>
            <li style="margin-bottom: 0.5rem;">
                <strong style="color: #e2e8f0;">Detection Rate</strong> - Must catch 100% of major stress events to be included. 
                Curve inversion (33% detection) was excluded despite theoretical importance.
            </li>
            <li style="margin-bottom: 0.5rem;">
                <strong style="color: #e2e8f0;">Independence</strong> - Indicators measuring different risk dimensions get priority. 
                VIX (fear), credit (default risk), and USD (flight to safety) capture distinct signals.
            </li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    weights_data = []
    for key, weight in sorted(WEIGHTS.items(), key=lambda x: x[1], reverse=True):
        info = INDICATOR_INFO.get(key, {})
        weights_data.append({
            "Indicator": info.get("name", key),
            "Weight": f"{weight*100:.0f}%",
            "Signal Ratio": info.get("signal_ratio", "N/A"),
            "Why": info.get("weight_rationale", "Based on backtest performance")
        })
    
    st.dataframe(pd.DataFrame(weights_data), use_container_width=True, hide_index=True)
    
    # Non-linear model comparison section
    st.markdown("---")
    st.markdown("## Non-Linear Model Comparison")
    
    st.markdown("""
    We tested multiple composite score formulations to see if non-linear relationships 
    improve fast market detection. Each model was backtested against the 6 historical stress events.
    """)
    
    # Model descriptions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Linear (Baseline)</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Simple weighted average of indicator percentiles.
                Each indicator contributes independently.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Convex</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Power transformation (pct^1.4) amplifies extreme readings.
                85th percentile matters more than 50th.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Regime</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Different weights for calm vs. stress regimes.
                Emphasize credit (leading) when calm, VIX (confirming) in stress.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0 0 0.5rem 0;">Convergence</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Multiplier when multiple indicators spike together.
                3+ elevated indicators = compounding risk (diversification fails).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
            <h4 style="color: #3b82f6; margin: 0 0 0.5rem 0;">Hybrid (Recommended)</h4>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                Combines convex + regime + convergence + lead-lag.
                Most sensitive to stress but higher false positive rate.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Model Performance")
    
    # Model comparison table
    model_data = [
        {"Model": "Hybrid", "Detection": "100%", "Signal Ratio": "1.82x", "Lead Days": "0.8", "Precision": "42%", "FP Rate": "9.7%"},
        {"Model": "Convergence", "Detection": "100%", "Signal Ratio": "1.73x", "Lead Days": "1.8", "Precision": "40%", "FP Rate": "10.8%"},
        {"Model": "Convex", "Detection": "100%", "Signal Ratio": "1.53x", "Lead Days": "0.0", "Precision": "70%", "FP Rate": "0.8%"},
        {"Model": "Linear", "Detection": "100%", "Signal Ratio": "1.47x", "Lead Days": "0.0", "Precision": "70%", "FP Rate": "1.1%"},
        {"Model": "Regime", "Detection": "100%", "Signal Ratio": "1.45x", "Lead Days": "0.0", "Precision": "71%", "FP Rate": "0.9%"},
    ]
    
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
    
    st.markdown("### Interpretation")
    
    st.markdown("""
    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 1rem;">
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0 0 0.75rem 0;">
            <strong style="color: #3b82f6;">Hybrid model</strong> shows the best signal ratio (1.82x vs 1.47x baseline), 
            meaning it distinguishes stress from calm markets more clearly. Peak scores during crises reach 95-99 
            vs. 71-73 for linear.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0 0 0.75rem 0;">
            <strong style="color: #f59e0b;">Tradeoff:</strong> Higher sensitivity comes with more false positives 
            (42% precision vs 70%). The model fires more often outside stress events.
        </p>
        <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
            <strong style="color: #10b981;">Recommendation:</strong> Use <strong>linear</strong> for daily monitoring 
            (fewer false alarms). Switch to <strong>hybrid</strong> lens when linear crosses 50+ to amplify early 
            warning signals.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main() -> None:
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="Market Stress Monitor",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Global styles
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            .stApp { background-color: #0f172a; font-family: 'Inter', sans-serif; }
            .stMarkdown, .stText, p, span, label { color: #e2e8f0; }
            h1, h2, h3, h4 { color: #f1f5f9 !important; font-weight: 600 !important; }
            hr { border-color: #334155; }
            .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
            .stTabs [data-baseweb="tab"] { 
                color: #94a3b8; 
                font-weight: 500;
                padding: 0.5rem 0;
            }
            .stTabs [aria-selected="true"] { 
                color: #3b82f6 !important;
                border-bottom-color: #3b82f6 !important;
            }
            .stDataFrame { background: #1e293b; }
            #MainMenu, footer, header { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Header
    st.markdown(
        """<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
            <div>
                <h1 style="margin: 0; font-size: 1.5rem; color: #f1f5f9;">Market Stress Monitor</h1>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">Asset Management Risk Dashboard</div>
            </div>
            <div style="color: #64748b; font-size: 0.7rem; text-align: right;">
                Data: FRED + Yahoo Finance<br>Refresh: Daily EOD
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    
    # Period selector
    period_options = {
        "90 Days": 90,
        "180 Days": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825,
        "All Available": 5000,
    }
    
    col_period, col_spacer = st.columns([1, 4])
    with col_period:
        selected_period = st.selectbox(
            "History Period",
            options=list(period_options.keys()),
            index=0,
            label_visibility="collapsed",
        )
    history_days = period_options[selected_period]
    
    # Load data
    with st.spinner("Loading..."):
        calc = IndicatorCalculator()
        result = calc.calculate()
        history = calc.get_history(days=history_days)
    
    if result is None:
        st.error("No data available. Run: python -m fast_market_dashboard.data.fred_fetcher --all")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Indicators", "Backtest"])
    
    with tab1:
        render_dashboard_tab(calc, result, history, history_days)
    
    with tab2:
        render_indicators_tab(result)
    
    with tab3:
        render_backtest_tab()


if __name__ == "__main__":
    main()
