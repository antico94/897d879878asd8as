import pandas as pd

from ReportGeneration.ReportUtils.html_utils import embed_image
from ReportGeneration.ReportUtils.chart_utils import (
    generate_equity_curve_chart,
    generate_drawdown_chart,
    generate_monthly_returns_chart,
    generate_balance_comparison_chart
)


def generate_equity_section(backtest_results):
    """Generate the equity curve and drawdown section."""
    # Extract equity data
    equity_data_list = backtest_results.get('equity_curve', [])

    # Convert to DataFrame
    if equity_data_list:
        equity_data_df = pd.DataFrame(equity_data_list)
        if 'timestamp' in equity_data_df.columns:
            equity_data_df.set_index('timestamp', inplace=True)
        equity_data = equity_data_df
    else:
        equity_data = pd.DataFrame() # Or handle empty case as needed

    # Create charts
    equity_chart_path = generate_equity_curve_chart(equity_data, backtest_results.get('initial_balance', 10000))
    drawdown_chart_path = generate_drawdown_chart(equity_data)

    # Generate monthly returns chart if the data is available
    monthly_returns = backtest_results.get('monthly_returns', {})
    monthly_returns_chart_path = None

    if monthly_returns:
        monthly_returns_chart_path = generate_monthly_returns_chart(monthly_returns)

    # Create comparison chart if baseline data is available
    baseline_returns = backtest_results.get('baseline_returns', None)
    comparison_chart_path = None

    if baseline_returns:
        comparison_chart_path = generate_balance_comparison_chart(backtest_results, baseline_returns)

    # Extract metrics for context
    metrics = backtest_results.get('metrics', {})
    max_drawdown_pct = metrics.get('max_drawdown_pct', 0)
    return_pct = metrics.get('return_pct', 0)

    # Embed charts
    equity_chart_html = ""
    if equity_chart_path:
        embedded_img = embed_image(equity_chart_path)
        equity_chart_html = f"""
        <div class="chart-container">
            <h3>Equity Curve</h3>
            <img src="{embedded_img}" alt="Equity Curve">
            <div class="card-explanation">
                <p>This chart shows how the account value changes over time during the backtest period. Steady upward slopes indicate consistent profitability.</p>
            </div>
        </div>
        """

    drawdown_chart_html = ""
    if drawdown_chart_path:
        embedded_img = embed_image(drawdown_chart_path)
        drawdown_chart_html = f"""
        <div class="chart-container">
            <h3>Drawdown</h3>
            <img src="{embedded_img}" alt="Drawdown Chart">
            <div class="card-explanation">
                <p>This chart shows the percentage decline from previous peak account value. The maximum drawdown of {max_drawdown_pct:.2f}% represents the largest peak-to-trough decline.</p>
            </div>
        </div>
        """

    monthly_returns_html = ""
    if monthly_returns_chart_path:
        embedded_img = embed_image(monthly_returns_chart_path)
        monthly_returns_html = f"""
        <div class="chart-container">
            <h3>Monthly Returns</h3>
            <img src="{embedded_img}" alt="Monthly Returns">
            <div class="card-explanation">
                <p>This chart shows the monthly returns of the strategy. Green bars represent profitable months, while red bars represent losing months.</p>
            </div>
        </div>
        """

    comparison_chart_html = ""
    if comparison_chart_path:
        embedded_img = embed_image(comparison_chart_path)
        comparison_chart_html = f"""
        <div class="chart-container">
            <h3>Strategy vs. Baseline Comparison</h3>
            <img src="{embedded_img}" alt="Strategy vs. Baseline">
            <div class="card-explanation">
                <p>This chart compares the strategy's performance to a baseline (typically a buy-and-hold approach). It helps evaluate whether the strategy adds value compared to simpler approaches.</p>
            </div>
        </div>
        """

    # Create recommendations based on equity and drawdown
    equity_recommendations = []

    # Return consistency
    if monthly_returns:
        positive_months = sum(1 for return_val in monthly_returns.values() if return_val > 0)
        total_months = len(monthly_returns)
        if total_months > 0:
            positive_month_pct = positive_months / total_months * 100

            if positive_month_pct > 70:
                equity_recommendations.append(
                    f"The strategy shows high consistency with {positive_month_pct:.1f}% of months being profitable.")
            elif positive_month_pct < 50:
                equity_recommendations.append(
                    f"The strategy has more losing months ({100 - positive_month_pct:.1f}%) than winning months, suggesting inconsistent performance.")

    # Drawdown concerns
    if max_drawdown_pct > 30:
        equity_recommendations.append(
            f"The maximum drawdown of {max_drawdown_pct:.2f}% is significant. Consider implementing tighter risk controls or stopping trading during adverse market conditions.")
    elif max_drawdown_pct > 15:
        equity_recommendations.append(
            f"The maximum drawdown of {max_drawdown_pct:.2f}% is moderate. Ensure you have sufficient psychological tolerance and capital to handle such drawdowns.")
    else:
        equity_recommendations.append(
            f"The maximum drawdown of {max_drawdown_pct:.2f}% is well-controlled, suggesting good risk management.")

    # Recovery after drawdowns
    equity_recommendations.append(
        "Examine the equity curve for prolonged flat periods, which may indicate the strategy struggling to recover after drawdowns.")

    # Overall growth
    if return_pct > 20:
        equity_recommendations.append(
            f"The overall return of {return_pct:.2f}% demonstrates strong growth. Verify that leverage or risk settings are appropriate and not excessive.")

    # Create recommendations HTML
    recommendations_html = ""
    if equity_recommendations:
        recommendations_html = """
        <div class="recommendations">
            <h3>Equity Analysis Insights</h3>
            <ul>
        """
        for rec in equity_recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
            </ul>
        </div>
        """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Equity and Drawdown Analysis</div>
        <div class="card-body">
            {equity_chart_html}

            {drawdown_chart_html}

            {monthly_returns_html}

            {comparison_chart_html}

            {recommendations_html}
        </div>
    </div>
    """

    return html