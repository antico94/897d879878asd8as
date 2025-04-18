import os
import base64
from pathlib import Path


def embed_image(image_path):
    if not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    file_ext = os.path.splitext(image_path)[1].lower()[1:]
    if file_ext == 'jpg':
        file_ext = 'jpeg'

    return f"data:image/{file_ext};base64,{encoded_image}"


def get_badge_class(value, thresholds):
    """Return the appropriate badge class based on value and thresholds."""
    if value >= thresholds['good']:
        return "badge-success"
    elif value >= thresholds['average']:
        return "badge-warning"
    else:
        return "badge-danger"


def get_text_class(value, thresholds):
    """Return the appropriate text class based on value and thresholds."""
    if value >= thresholds['good']:
        return "metric-good"
    elif value >= thresholds['average']:
        return "metric-average"
    else:
        return "metric-poor"


def create_html_report(sections, timestamp):
    """Create a complete HTML report from the provided sections."""
    css = _get_css()

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Gold Trading Backtest Report</title>
        <style>
        {css}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Gold Trading Backtest Report</h1>
                <p>Comprehensive analysis of backtesting results and trading performance</p>
                <p><small>Generated on: {timestamp}</small></p>
            </div>

            {sections['summary']}

            {sections['model_info']}

            {sections['performance']}

            {sections['equity']}

            {sections['trades']}

            {sections['accuracy']}

            {sections.get('confidence_risk', '')}

            <footer>
                <p>Gold Trading Backtest Report | Generated by Gold Trading Bot</p>
            </footer>
        </div>

        <script>
            // Add interactive features
            document.addEventListener('DOMContentLoaded', function() {{
                // Make tooltips work
                const tooltips = document.querySelectorAll('.tooltip');
                tooltips.forEach(tooltip => {{
                    tooltip.addEventListener('mouseenter', function() {{
                        const tooltiptext = this.querySelector('.tooltiptext');
                        tooltiptext.style.visibility = 'visible';
                        tooltiptext.style.opacity = '1';
                    }});
                    tooltip.addEventListener('mouseleave', function() {{
                        const tooltiptext = this.querySelector('.tooltiptext');
                        tooltiptext.style.visibility = 'hidden';
                        tooltiptext.style.opacity = '0';
                    }});
                }});

                // Initialize tabs if they exist
                const tabLinks = document.querySelectorAll('.tab-link');
                tabLinks.forEach(link => {{
                    link.addEventListener('click', function(e) {{
                        e.preventDefault();

                        // Get the tab content id
                        const tabId = this.getAttribute('data-tab');

                        // Hide all tab contents
                        document.querySelectorAll('.tab-content').forEach(content => {{
                            content.style.display = 'none';
                        }});

                        // Remove active class from all tab links
                        document.querySelectorAll('.tab-link').forEach(link => {{
                            link.classList.remove('active');
                        }});

                        // Show the selected tab content
                        document.getElementById(tabId).style.display = 'block';

                        // Add active class to the clicked tab link
                        this.classList.add('active');
                    }});
                }});

                // Activate the first tab if it exists
                const firstTab = document.querySelector('.tab-link');
                if (firstTab) {{
                    firstTab.click();
                }}
            }});
        </script>
    </body>
    </html>
    """

    return html


def _get_css():
    """Get the CSS styles for the report."""
    return """
    :root {
        --primary-color: #1a5276;
        --secondary-color: #f39c12;
        --background-color: #f8f9fa;
        --card-bg: white;
        --text-color: #333;
        --border-color: #ddd;
        --success-color: #27ae60;
        --warning-color: #e67e22;
        --danger-color: #c0392b;
        --info-color: #3498db;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--background-color);
        margin: 0;
        padding: 0;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }

    .header {
        background-color: var(--primary-color);
        color: white;
        padding: 20px;
        text-align: center;
        border-radius: 8px 8px 0 0;
        margin-bottom: 30px;
    }

    .header h1 {
        margin: 0;
        font-size: 28px;
    }

    .header p {
        margin: 10px 0 0;
        opacity: 0.8;
    }

    .card {
        background-color: var(--card-bg);
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        overflow: hidden;
    }

    .card-header {
        background-color: var(--primary-color);
        color: white;
        padding: 15px 20px;
        font-weight: bold;
        font-size: 18px;
        border-bottom: 1px solid var(--border-color);
    }

    .card-body {
        padding: 20px;
    }

    .card-explanation {
        background-color: rgba(52, 152, 219, 0.1);
        border-left: 4px solid var(--info-color);
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 8px 8px 0;
    }

    .metric {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }

    .metric-label {
        width: 250px;
        font-weight: bold;
    }

    .metric-value {
        flex-grow: 1;
    }

    .metric-good {
        color: var(--success-color);
        font-weight: bold;
    }

    .metric-average {
        color: var(--warning-color);
        font-weight: bold;
    }

    .metric-poor {
        color: var(--danger-color);
        font-weight: bold;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }

    table, th, td {
        border: 1px solid var(--border-color);
    }

    th {
        background-color: #f2f2f2;
        padding: 12px;
        text-align: left;
    }

    td {
        padding: 10px;
    }

    tr:nth-child(even) {
        background-color: #f9f9f9;
    }

    .chart-container {
        margin: 20px 0;
        max-width: 100%;
        text-align: center;
    }

    .chart-container img {
        max-width: 100%;
        height: auto;
        border: 1px solid var(--border-color);
        border-radius: 4px;
    }

    .summary-box {
        background-color: #f8f9fa;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
    }

    .summary-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: var(--primary-color);
    }

    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-weight: normal;
        font-size: 14px;
        line-height: 1.4;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        margin-left: 8px;
    }

    .badge-success {
        background-color: var(--success-color);
        color: white;
    }

    .badge-warning {
        background-color: var(--warning-color);
        color: white;
    }

    .badge-danger {
        background-color: var(--danger-color);
        color: white;
    }

    .recommendations {
        background-color: #ebf5fb;
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        margin: 20px 0;
        border-radius: 0 8px 8px 0;
    }

    .recommendations h3 {
        margin-top: 0;
        color: var(--primary-color);
    }

    .recommendations ul {
        margin-bottom: 0;
    }

    .tab-container {
        margin-bottom: 20px;
    }

    .tab-nav {
        display: flex;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 15px;
    }

    .tab-link {
        padding: 10px 20px;
        cursor: pointer;
        background-color: #f2f2f2;
        border: none;
        border-radius: 8px 8px 0 0;
        margin-right: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }

    .tab-link.active {
        background-color: var(--primary-color);
        color: white;
    }

    .tab-content {
        display: none;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 0 0 8px 8px;
        background-color: white;
    }

    .progress-bar-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 8px 0;
    }

    .progress-bar {
        height: 20px;
        border-radius: 4px;
        text-align: center;
        line-height: 20px;
        color: white;
        font-weight: bold;
    }

    .trade-detail {
        background-color: #ebf5fb;
        border: 1px solid #3498db;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    .trade-profitable {
        border-left: 5px solid var(--success-color);
    }

    .trade-unprofitable {
        border-left: 5px solid var(--danger-color);
    }

    .trade-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .trade-id {
        font-weight: bold;
        font-size: 16px;
    }

    .trade-profit {
        font-weight: bold;
    }

    .profit-positive {
        color: var(--success-color);
    }

    .profit-negative {
        color: var(--danger-color);
    }

    .trade-details-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        grid-gap: 10px;
    }

    .trade-detail-item {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 4px;
    }

    .trade-detail-label {
        font-weight: bold;
        font-size: 12px;
        color: #666;
    }

    .trade-detail-value {
        font-size: 14px;
    }

    footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        color: #777;
        font-size: 14px;
        border-top: 1px solid var(--border-color);
    }

    .col-md-6 {
        width: 48%;
        display: inline-block;
        vertical-align: top;
        margin-right: 2%;
    }

    .row {
        display: flex;
        flex-wrap: wrap;
        margin: 0 -10px;
    }

    .col {
        flex: 1;
        padding: 0 10px;
    }

    @media (max-width: 768px) {
        .col-md-6 {
            width: 100%;
            margin-right: 0;
        }

        .metric {
            flex-direction: column;
            align-items: flex-start;
        }

        .metric-label {
            width: 100%;
            margin-bottom: 5px;
        }

        .row {
            flex-direction: column;
        }
    }
    """