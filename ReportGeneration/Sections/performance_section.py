from ReportGeneration.ReportUtils.html_utils import get_badge_class, get_text_class


def generate_performance_section(backtest_results):
    """Generate the performance metrics section."""
    metrics = backtest_results.get('metrics', {})

    # Extract key metrics
    initial_balance = backtest_results.get('initial_balance', 10000)
    final_balance = metrics.get('final_balance', initial_balance)
    net_profit = metrics.get('net_profit', final_balance - initial_balance)
    return_pct = metrics.get('return_pct', (final_balance / initial_balance - 1) * 100)
    annualized_return = metrics.get('annualized_return', 0)

    # Risk metrics
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    sortino_ratio = metrics.get('sortino_ratio', 0)
    max_drawdown = metrics.get('max_drawdown', 0)
    max_drawdown_pct = metrics.get('max_drawdown_pct', 0)

    # Trade metrics
    total_trades = metrics.get('total_trades', 0)
    winning_trades = metrics.get('winning_trades', 0)
    losing_trades = metrics.get('losing_trades', 0)
    win_rate = metrics.get('win_rate', 0)
    profit_factor = metrics.get('profit_factor', 0)

    # Average metrics
    avg_win = metrics.get('avg_win', 0)
    avg_loss = metrics.get('avg_loss', 0)
    avg_trade = metrics.get('avg_profit_per_trade', 0)

    # Define thresholds for color coding
    return_thresholds = {'good': 10, 'average': 0}
    win_rate_thresholds = {'good': 0.55, 'average': 0.45}
    profit_factor_thresholds = {'good': 1.5, 'average': 1.0}
    sharpe_thresholds = {'good': 1.0, 'average': 0.5}

    # Create metrics HTML
    key_metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Initial Balance:</div>
        <div class="metric-value">${initial_balance:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Final Balance:</div>
        <div class="metric-value">${final_balance:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Net Profit:</div>
        <div class="metric-value {get_text_class(net_profit, {'good': 1, 'average': 0})}">${net_profit:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Return:</div>
        <div class="metric-value {get_text_class(return_pct, return_thresholds)}">{return_pct:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Annualized Return:</div>
        <div class="metric-value {get_text_class(annualized_return, return_thresholds)}">{annualized_return:.2f}%</div>
    </div>
    """

    risk_metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Sharpe Ratio:</div>
        <div class="metric-value {get_text_class(sharpe_ratio, sharpe_thresholds)}">{sharpe_ratio:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Sortino Ratio:</div>
        <div class="metric-value {get_text_class(sortino_ratio, sharpe_thresholds)}">{sortino_ratio:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Maximum Drawdown:</div>
        <div class="metric-value {get_text_class(-max_drawdown_pct, {'good': -5, 'average': -15})}">${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)</div>
    </div>
    """

    trade_metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Total Trades:</div>
        <div class="metric-value">{total_trades}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Winning Trades:</div>
        <div class="metric-value">{winning_trades} ({winning_trades / total_trades * 100:.1f}% of total)</div>
    </div>
    <div class="metric">
        <div class="metric-label">Losing Trades:</div>
        <div class="metric-value">{losing_trades} ({losing_trades / total_trades * 100:.1f}% of total)</div>
    </div>
    <div class="metric">
        <div class="metric-label">Win Rate:</div>
        <div class="metric-value {get_text_class(win_rate, win_rate_thresholds)}">{win_rate:.2%}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Profit Factor:</div>
        <div class="metric-value {get_text_class(profit_factor, profit_factor_thresholds)}">{profit_factor:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Winning Trade:</div>
        <div class="metric-value">${avg_win:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Losing Trade:</div>
        <div class="metric-value">-${abs(avg_loss):.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Trade P/L:</div>
        <div class="metric-value {get_text_class(avg_trade, {'good': 1, 'average': 0})}">${avg_trade:.2f}</div>
    </div>
    """

    # Add explanations for metrics
    metrics_explanation = """
    <div class="card-explanation">
        <p><strong>Key Performance Metrics Explained:</strong></p>
        <ul>
            <li><strong>Net Profit</strong>: The total profit or loss generated by the strategy over the backtest period.</li>
            <li><strong>Return</strong>: The percentage return on the initial investment.</li>
            <li><strong>Annualized Return</strong>: The return normalized to a yearly rate, useful for comparing strategies tested over different time periods.</li>
            <li><strong>Win Rate</strong>: The percentage of trades that resulted in a profit.</li>
            <li><strong>Profit Factor</strong>: The ratio of gross profits to gross losses. A value above 1.0 indicates a profitable system.</li>
            <li><strong>Sharpe Ratio</strong>: A measure of risk-adjusted return. Higher is better, with values above 1.0 generally considered good.</li>
            <li><strong>Maximum Drawdown</strong>: The largest peak-to-trough decline in account value, both in dollar terms and as a percentage.</li>
        </ul>
    </div>
    """

    # Add recommendations based on the metrics
    recommendations = []

    # Overall profitability
    if return_pct <= 0:
        recommendations.append(
            "The strategy is not profitable. Consider revising the entry/exit conditions or risk parameters.")
    elif return_pct < 5:
        recommendations.append(
            "The strategy is marginally profitable. Consider optimizing for improved returns while keeping risk in check.")
    elif return_pct > 25:
        recommendations.append(
            "The strategy shows strong profitability. Verify that assumptions are realistic and not overfit to historical data.")

    # Win rate and profit factor
    if win_rate < 0.4:
        recommendations.append(
            "The low win rate requires significantly larger winning trades than losing trades to remain profitable.")
    elif win_rate > 0.6:
        recommendations.append(
            "High win rate indicates good signal quality. Potential to increase position sizing for larger overall returns.")

    if profit_factor < 1.0:
        recommendations.append(
            "Profit factor below 1.0 indicates the strategy is losing money. Immediate review recommended.")
    elif profit_factor > 2.0:
        recommendations.append("Strong profit factor indicates an excellent ratio of gains to losses.")

    # Risk metrics
    if max_drawdown_pct > 20:
        recommendations.append(
            "Large maximum drawdown may make the strategy challenging to trade psychologically. Consider reducing position sizes.")

    if sharpe_ratio < 0.5:
        recommendations.append(
            "Low Sharpe ratio indicates poor risk-adjusted returns. The strategy may not be compensating adequately for the risk taken.")
    elif sharpe_ratio > 1.5:
        recommendations.append("High Sharpe ratio indicates excellent risk-adjusted returns.")

    # Create recommendations HTML
    recommendations_html = ""
    if recommendations:
        recommendations_html = """
        <div class="recommendations">
            <h3>Performance Insights</h3>
            <ul>
        """
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
            </ul>
        </div>
        """

    # Build the complete section with tabs
    html = f"""
    <div class="card">
        <div class="card-header">Performance Metrics</div>
        <div class="card-body">
            <div class="tab-container">
                <div class="tab-nav">
                    <button class="tab-link" data-tab="key-metrics-tab">Key Metrics</button>
                    <button class="tab-link" data-tab="risk-metrics-tab">Risk Metrics</button>
                    <button class="tab-link" data-tab="trade-metrics-tab">Trade Metrics</button>
                </div>

                <div id="key-metrics-tab" class="tab-content">
                    <h3>Key Performance Metrics</h3>
                    {key_metrics_html}
                </div>

                <div id="risk-metrics-tab" class="tab-content">
                    <h3>Risk Metrics</h3>
                    {risk_metrics_html}
                </div>

                <div id="trade-metrics-tab" class="tab-content">
                    <h3>Trade Metrics</h3>
                    {trade_metrics_html}
                </div>
            </div>

            {metrics_explanation}

            {recommendations_html}
        </div>
    </div>
    """

    return html