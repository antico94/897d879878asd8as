import pandas as pd
from ReportGeneration.ReportUtils.html_utils import embed_image, get_text_class
from ReportGeneration.ReportUtils.chart_utils import (
    generate_win_loss_chart,
    generate_trade_duration_chart,
    generate_trade_size_chart,
    generate_win_rate_by_day_chart,
    generate_win_rate_by_hour_chart
)


def generate_trades_section(backtest_results):
    """Generate the trades analysis section."""
    # Extract trades data
    trades = backtest_results.get('trades', [])

    if not trades:
        return '<div class="card"><div class="card-header">Trades Analysis</div><div class="card-body"><p>No trades data available for analysis.</p></div></div>'

    # Generate charts
    win_loss_chart_path, cumulative_pnl_path = generate_win_loss_chart(trades)
    duration_chart_path = generate_trade_duration_chart(trades)
    size_chart_path = generate_trade_size_chart(trades)
    day_win_rate_path = generate_win_rate_by_day_chart(trades)
    hour_win_rate_path = generate_win_rate_by_hour_chart(trades)

    # Calculate trade statistics
    total_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if trade.get('profit_loss', 0) > 0)
    losing_trades = total_trades - profitable_trades
    win_rate = profitable_trades / total_trades if total_trades > 0 else 0

    # Calculate trade sizes
    position_sizes = [trade.get('position_size', 0) for trade in trades]
    avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
    max_position_size = max(position_sizes) if position_sizes else 0

    # Calculate trade durations
    durations = []
    for trade in trades:
        if 'trade_duration' in trade:
            durations.append(trade['trade_duration'])
        elif 'entry_time' in trade and 'exit_time' in trade:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            duration_hours = (exit_time - entry_time).total_seconds() / 3600
            durations.append(duration_hours)

    avg_duration = sum(durations) / len(durations) if durations else 0
    max_duration = max(durations) if durations else 0

    # Calculate profit/loss statistics
    profits = [trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) > 0]
    losses = [trade.get('profit_loss', 0) for trade in trades if trade.get('profit_loss', 0) <= 0]

    avg_profit = sum(profits) / len(profits) if profits else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    max_profit = max(profits) if profits else 0
    max_loss = min(losses) if losses else 0

    # Create trade summary metrics
    trade_summary_html = f"""
    <div class="metric">
        <div class="metric-label">Total Trades:</div>
        <div class="metric-value">{total_trades}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Profitable Trades:</div>
        <div class="metric-value">{profitable_trades} ({profitable_trades / total_trades * 100:.1f}%)</div>
    </div>
    <div class="metric">
        <div class="metric-label">Losing Trades:</div>
        <div class="metric-value">{losing_trades} ({losing_trades / total_trades * 100:.1f}%)</div>
    </div>
    <div class="metric">
        <div class="metric-label">Win Rate:</div>
        <div class="metric-value {get_text_class(win_rate, {'good': 0.55, 'average': 0.45})}">{win_rate:.2%}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Profit:</div>
        <div class="metric-value">${avg_profit:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Loss:</div>
        <div class="metric-value">-${abs(avg_loss):.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Largest Profit:</div>
        <div class="metric-value">${max_profit:.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Largest Loss:</div>
        <div class="metric-value">-${abs(max_loss):.2f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Position Size:</div>
        <div class="metric-value">{avg_position_size:.2f} lots</div>
    </div>
    <div class="metric">
        <div class="metric-label">Average Trade Duration:</div>
        <div class="metric-value">{avg_duration:.2f} hours</div>
    </div>
    """

    # Create chart sections
    charts_html = ""

    # Add win/loss distribution chart
    if win_loss_chart_path:
        embedded_img = embed_image(win_loss_chart_path)
        charts_html += f"""
        <div class="chart-container">
            <h3>Profit/Loss Distribution</h3>
            <img src="{embedded_img}" alt="Profit/Loss Distribution">
            <div class="card-explanation">
                <p>This chart shows the distribution of profits and losses across all trades. A healthy distribution should be positively skewed (more bars on the right side).</p>
            </div>
        </div>
        """

    # Add cumulative P&L chart
    if cumulative_pnl_path:
        embedded_img = embed_image(cumulative_pnl_path)
        charts_html += f"""
        <div class="chart-container">
            <h3>Cumulative Profit/Loss</h3>
            <img src="{embedded_img}" alt="Cumulative Profit/Loss">
            <div class="card-explanation">
                <p>This chart shows the cumulative profit/loss over time. Ideally, it should show a steady upward slope without prolonged flat or declining periods.</p>
            </div>
        </div>
        """

    # Add trade duration chart
    if duration_chart_path:
        embedded_img = embed_image(duration_chart_path)
        charts_html += f"""
        <div class="chart-container">
            <h3>Trade Duration Distribution</h3>
            <img src="{embedded_img}" alt="Trade Duration Distribution">
            <div class="card-explanation">
                <p>This chart shows how long trades typically last. The average trade duration is {avg_duration:.2f} hours.</p>
            </div>
        </div>
        """

    # Add win rate by day chart
    if day_win_rate_path:
        embedded_img = embed_image(day_win_rate_path)
        charts_html += f"""
        <div class="chart-container">
            <h3>Win Rate by Day of Week</h3>
            <img src="{embedded_img}" alt="Win Rate by Day">
            <div class="card-explanation">
                <p>This chart shows how the strategy performs on different days of the week. Use this to identify which days might be better for trading.</p>
            </div>
        </div>
        """

    # Add win rate by hour chart
    if hour_win_rate_path:
        embedded_img = embed_image(hour_win_rate_path)
        charts_html += f"""
        <div class="chart-container">
            <h3>Win Rate by Hour of Day</h3>
            <img src="{embedded_img}" alt="Win Rate by Hour">
            <div class="card-explanation">
                <p>This chart shows how the strategy performs during different hours of the day. Use this to identify the best trading hours.</p>
            </div>
        </div>
        """

    # Create table of recent trades
    recent_trades_html = """
    <h3>Recent Trades</h3>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th>Entry Time</th>
                <th>Direction</th>
                <th>Position Size</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>Profit/Loss</th>
                <th>Duration</th>
            </tr>
        </thead>
        <tbody>
    """

    # Show up to 10 recent trades
    recent_trades = sorted(trades, key=lambda x: x.get('entry_time', ''), reverse=True)[:10]

    for trade in recent_trades:
        trade_id = trade.get('id', '')
        entry_time = trade.get('entry_time', '')
        direction = trade.get('direction', '')
        position_size = trade.get('position_size', 0)
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        profit_loss = trade.get('profit_loss', 0)

        # Calculate duration string
        duration_str = ''
        if 'trade_duration' in trade:
            hours = trade['trade_duration']
            if hours < 1:
                duration_str = f"{hours * 60:.0f}m"
            else:
                duration_str = f"{hours:.1f}h"
        elif 'entry_time' in trade and 'exit_time' in trade:
            entry_time_obj = pd.to_datetime(entry_time)
            exit_time_obj = pd.to_datetime(trade['exit_time'])
            duration = exit_time_obj - entry_time_obj
            hours = duration.total_seconds() / 3600
            if hours < 1:
                duration_str = f"{duration.total_seconds() / 60:.0f}m"
            else:
                duration_str = f"{hours:.1f}h"

        # Add row with appropriate color for profit/loss
        profit_class = "profit-positive" if profit_loss > 0 else "profit-negative"
        recent_trades_html += f"""
        <tr>
            <td>{trade_id}</td>
            <td>{entry_time}</td>
            <td>{direction}</td>
            <td>{position_size:.2f}</td>
            <td>${entry_price:.2f}</td>
            <td>${exit_price:.2f}</td>
            <td class="{profit_class}">${profit_loss:.2f}</td>
            <td>{duration_str}</td>
        </tr>
        """

    recent_trades_html += """
        </tbody>
    </table>
    """

    # Create trade insights and recommendations
    trade_insights = []

    # Win rate insights
    if win_rate < 0.4:
        trade_insights.append(
            f"The win rate of {win_rate:.2%} is relatively low. The strategy relies on winning trades being significantly larger than losing trades.")
    elif win_rate > 0.6:
        trade_insights.append(f"The high win rate of {win_rate:.2%} suggests the strategy has good predictive power.")

    # Risk-reward insights
    reward_risk_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
    if reward_risk_ratio < 1.0 and win_rate < 0.5:
        trade_insights.append(
            f"The reward-to-risk ratio ({reward_risk_ratio:.2f}) is unfavorable combined with a win rate below 50%. This makes consistent profitability challenging.")
    elif reward_risk_ratio > 2.0:
        trade_insights.append(
            f"The strong reward-to-risk ratio ({reward_risk_ratio:.2f}) means winning trades are significantly larger than losing trades.")

    # Duration insights
    if durations:
        if avg_duration < 2:
            trade_insights.append(
                f"The average trade duration of {avg_duration:.2f} hours is quite short. This might indicate a high-frequency approach.")
        elif avg_duration > 48:
            trade_insights.append(
                f"The average trade duration of {avg_duration:.2f} hours (about {avg_duration / 24:.1f} days) suggests a longer-term approach.")

    # Day/hour trading insights
    if day_win_rate_path:
        trade_insights.append("Analyze the win rate by day chart to identify optimal trading days for this strategy.")

    if hour_win_rate_path:
        trade_insights.append("Consider focusing trading during hours that show consistently higher win rates.")

    # Create insights HTML
    insights_html = ""
    if trade_insights:
        insights_html = """
        <div class="recommendations">
            <h3>Trade Analysis Insights</h3>
            <ul>
        """
        for insight in trade_insights:
            insights_html += f"<li>{insight}</li>"
        insights_html += """
            </ul>
        </div>
        """

    # Build the complete section with tabs
    html = f"""
    <div class="card">
        <div class="card-header">Trade Analysis</div>
        <div class="card-body">
            <div class="tab-container">
                <div class="tab-nav">
                    <button class="tab-link" data-tab="trade-summary-tab">Trade Summary</button>
                    <button class="tab-link" data-tab="trade-charts-tab">Trade Charts</button>
                    <button class="tab-link" data-tab="recent-trades-tab">Recent Trades</button>
                </div>

                <div id="trade-summary-tab" class="tab-content">
                    <h3>Trade Statistics</h3>
                    {trade_summary_html}
                </div>

                <div id="trade-charts-tab" class="tab-content">
                    {charts_html}
                </div>

                <div id="recent-trades-tab" class="tab-content">
                    {recent_trades_html}
                </div>
            </div>

            {insights_html}
        </div>
    </div>
    """

    return html