"""Update HTML with extended chart data."""
import json
import re

# Read the chart data
with open('chart_data.json', 'r') as f:
    data = json.load(f)

# Read the HTML
with open('dist/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

# Create the new CHART_DATA block
chart_data_js = f'''        // Chart data with multiple models
        const CHART_DATA = {{
            dates: {json.dumps(data['dates'])},
            sp500: {json.dumps(data['sp500'])},
            models: {{
                linear: {json.dumps(data['models']['linear'])},
                convex: {json.dumps(data['models']['convex'])},
                convergence: {json.dumps(data['models']['convergence'])},
                hybrid: {json.dumps(data['models']['hybrid'])}
            }}
        }};'''

# Find and replace the CHART_DATA section
pattern = r'        // Chart data with multiple models\s+const CHART_DATA = \{[^;]+\};'
html = re.sub(pattern, chart_data_js, html, flags=re.DOTALL)

# Write back
with open('dist/index.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Updated HTML with {len(data['dates'])} days of data")
