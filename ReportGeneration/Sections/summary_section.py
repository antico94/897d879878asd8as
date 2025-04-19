from ReportGeneration.ReportUtils.html_utils import get_text_class


def generate_summary_section(backtest_results):
    """Generate the executive summary section."""
    # Extract key metrics
    metrics = backtest_results.get('metrics', {})
    model_info = backtest_results.get('model_info', {})
    trades = backtest_results.get('trades', [])

    # Extract most important performance indicators
    initial_balance = backtest_results.get('initial_balance', 10000)
    final_balance = metrics.get('final_balance', initial_balance)
    net_profit = metrics.get('net_profit', final_balance - initial_balance)
    return_pct = metrics.get('return_pct', (final_balance / initial_balance - 1) * 100)
    win_rate = metrics.get('win_rate', 0)
    profit_factor = metrics.get('profit_factor', 0)
    max_drawdown_pct = metrics.get('max_drawdown_pct', 0)
    total_trades = metrics.get('total_trades', len(trades))

    # Define thresholds for color coding
    return_thresholds = {'good': 10, 'average': 0}
    win_rate_thresholds = {'good': 0.55, 'average': 0.45}
    profit_factor_thresholds = {'good': 1.5, 'average': 1.0}

    # Create summary card with key metrics
    summary_metrics = f"""
    <div class="metric">
        <div class="metric-label">Net Profit:</div>
        <div class="metric-value {get_text_class(net_profit, {'good': 1, 'average': 0})}">${net_profit:,.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Return:</div>
        <div class="metric-value {get_text_class(return_pct, return_thresholds)}">{return_pct:.2f}%</div>
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
        <div class="metric-label">Max Drawdown:</div>
        <div class="metric-value {get_text_class(-max_drawdown_pct, {'good': -5, 'average': -15})}">{max_drawdown_pct:.2f}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Total Trades:</div>
        <div class="metric-value">{total_trades}</div>
    </div>
    """

    # Identify key strengths and weaknesses
    strengths = []
    weaknesses = []

    # Assess overall profitability
    if return_pct > 20:
        strengths.append(f"Strong overall return ({return_pct:.2f}%)")
    elif return_pct <= 0:
        weaknesses.append("Strategy does not generate positive returns")

    # Assess win rate
    if win_rate > 0.6:
        strengths.append(f"High win rate ({win_rate:.2%})")
    elif win_rate < 0.4:
        weaknesses.append(f"Low win rate ({win_rate:.2%})")

    # Assess profit factor
    if profit_factor > 2:
        strengths.append(f"Excellent profit factor ({profit_factor:.2f})")
    elif profit_factor < 1:
        weaknesses.append("Profit factor below 1.0 indicates unprofitable strategy")

    # Assess drawdown
    if max_drawdown_pct < 10:
        strengths.append(f"Well-controlled drawdown ({max_drawdown_pct:.2f}%)")
    elif max_drawdown_pct > 25:
        weaknesses.append(f"Large maximum drawdown ({max_drawdown_pct:.2f}%)")

    # Assess trade count
    if total_trades < 20:
        weaknesses.append(f"Limited sample size of trades ({total_trades})")
    elif total_trades > 100:
        strengths.append(f"Good sample size of trades ({total_trades})")

    # Create HTML for strengths and weaknesses
    strengths_html = "<ul>"
    for strength in strengths:
        strengths_html += f"<li>{strength}</li>"
    strengths_html += "</ul>"

    weaknesses_html = "<ul>"
    for weakness in weaknesses:
        weaknesses_html += f"<li>{weakness}</li>"
    weaknesses_html += "</ul>"

    # Generate overall assessment
    overall_assessment = ""
    if return_pct > 15 and profit_factor > 1.5 and max_drawdown_pct < 20:
        overall_assessment = f"""
        <div class="metric-good">
            <p>This strategy demonstrates strong performance with a good balance of returns, reliability, and risk control.</p>
        </div>
        """
    elif return_pct > 0 and profit_factor > 1.0:
        overall_assessment = f"""
        <div class="metric-average">
            <p>This strategy shows moderate performance. It is profitable but may benefit from optimization to improve returns or reduce risk.</p>
        </div>
        """
    else:
        overall_assessment = f"""
        <div class="metric-poor">
            <p>This strategy requires significant improvement before it can be considered for live trading. Key issues should be addressed.</p>
        </div>
        """

    # Create recommendations
    recommendations = []

    # Add recommendations based on metrics
    if return_pct <= 0:
        recommendations.append("Revisit the trading strategy logic or model to achieve positive returns")
    elif return_pct < 5:
        recommendations.append("Consider optimizing the strategy to improve overall returns")

    if win_rate < 0.4 and profit_factor > 1:
        recommendations.append(
            "The low win rate but positive profit factor indicates a strategy that lets profits run but may experience long periods of small losses")

    if max_drawdown_pct > 20:
        recommendations.append("Implement tighter risk management to reduce maximum drawdown")

    if profit_factor < 1.2 and profit_factor > 1:
        recommendations.append(
            "The strategy is only marginally profitable. Consider increasing position size for winning trades or cutting losses faster")

    # Add general recommendations
    if return_pct > 0:
        recommendations.append("Perform additional testing on out-of-sample data to validate strategy robustness")
        recommendations.append("Consider forward testing in a demo account before live trading")

    # Create recommendations HTML
    recommendations_html = "<ul>"
    for rec in recommendations:
        recommendations_html += f"<li>{rec}</li>"
    recommendations_html += "</ul>"

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Executive Summary</div>
        <div class="card-body">
            <div class="row">
                <div class="col">
                    <h3>Key Performance Metrics</h3>
                    {summary_metrics}
                </div>

                <div class="col">
                    <h3>Overall Assessment</h3>
                    {overall_assessment}

                    <h4>Strengths</h4>
                    {strengths_html}

                    <h4>Areas for Improvement</h4>
                    {weaknesses_html}
                </div>
            </div>

            <div class="recommendations">
                <h3>Key Recommendations</h3>
                {recommendations_html}
            </div>
        </div>
    </div>
    """

    return html