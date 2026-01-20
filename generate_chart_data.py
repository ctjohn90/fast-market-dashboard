"""Generate chart data for HTML export."""
import json
import pandas as pd
from fast_market_dashboard.config import Settings
from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.indicators.calculator import IndicatorCalculator
from fast_market_dashboard.indicators.nonlinear import NonlinearCalculator

settings = Settings()
calculator = IndicatorCalculator(settings)
history = calculator.get_history(days=1825)

dates = [d.strftime('%Y-%m-%d') for d in history.index]
linear = [round(v, 1) for v in history['composite'].values]

nl = NonlinearCalculator()
convex_scores = []
convergence_scores = []
hybrid_scores = []

for idx, row in history.iterrows():
    indicator_pcts = {}
    for col in history.columns:
        if col != 'composite' and not pd.isna(row[col]):
            indicator_pcts[col] = row[col]
    
    if indicator_pcts:
        convex_scores.append(round(nl.calculate_convex(indicator_pcts), 1))
        convergence_scores.append(round(nl.calculate_convergence(indicator_pcts), 1))
        hybrid_scores.append(round(nl.calculate_hybrid(indicator_pcts), 1))
    else:
        convex_scores.append(linear[len(convex_scores)])
        convergence_scores.append(linear[len(convergence_scores)])
        hybrid_scores.append(linear[len(hybrid_scores)])

cache = DataCache(settings.db_path)
sp_df = cache.get_series('SP500')
sp500_returns = []
if not sp_df.empty:
    start_date = history.index[0]
    sp_aligned = sp_df[sp_df.index >= start_date]
    if not sp_aligned.empty:
        base_price = sp_aligned['value'].iloc[0]
        sp_aligned = sp_aligned.copy()
        sp_aligned['return'] = ((sp_aligned['value'] / base_price) - 1) * 100
        for d in history.index:
            if d in sp_aligned.index:
                sp500_returns.append(round(sp_aligned.loc[d, 'return'], 2))
            elif len(sp500_returns) > 0:
                sp500_returns.append(sp500_returns[-1])
            else:
                sp500_returns.append(0)

output = {
    'dates': dates,
    'sp500': sp500_returns,
    'models': {
        'linear': linear,
        'convex': convex_scores,
        'convergence': convergence_scores,
        'hybrid': hybrid_scores
    }
}

with open('chart_data.json', 'w') as f:
    json.dump(output, f)

print(f"Saved {len(dates)} days of data to chart_data.json")
