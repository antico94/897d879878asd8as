from ReportGeneration.ReportUtils.html_utils import get_text_class, embed_image


def generate_confidence_risk_section(backtest_results):
    """Generate the confidence vs risk-reward analysis section."""
    # Extract confidence-risk analysis data
    confidence_analysis = backtest_results.get('confidence_risk_analysis', {})

    if not confidence_analysis:
        return '<div class="card"><div class="card-header">Confidence vs Risk-Reward Analysis</div><div class="card-body"><p>No confidence-risk analysis data available.</p></div></div>'

    # Extract key data
    model_params = confidence_analysis.get('model_params', {})
    confidence_data = confidence_analysis.get('confidence_analysis', [])
    charts = confidence_analysis.get('charts', {})

    # Create the metrics table
    metrics_html = """
    <h3>Confidence Level Analysis</h3>
    <table>
        <thead>
            <tr>
                <th>Confidence</th>
                <th>Win Rate</th>
                <th>R-Multiple</th>
                <th>Optimal RR</th>
                <th>Trades</th>
                <th>Net P/L</th>
            </tr>
        </thead>
        <tbody>
    """

    for row in confidence_data:
        conf = row.get('confidence_group', 0)
        win_rate = row.get('win_mean', 0)
        r_multiple = row.get('r_multiple_mean', 0)
        optimal_rr = row.get('optimal_risk_reward', 0)
        count = int(row.get('profit_loss_count', 0))
        pnl = row.get('profit_loss_sum', 0)

        win_class = get_text_class(win_rate, {'good': 0.55, 'average': 0.45})
        r_class = get_text_class(r_multiple, {'good': 0.2, 'average': 0})
        pnl_class = get_text_class(pnl, {'good': 1, 'average': 0})

        metrics_html += f"""
        <tr>
            <td>{conf:.2f}</td>
            <td class="{win_class}">{win_rate:.2%}</td>
            <td class="{r_class}">{r_multiple:.2f}</td>
            <td>{optimal_rr:.2f}</td>
            <td>{count}</td>
            <td class="{pnl_class}">${pnl:.2f}</td>
        </tr>
        """

    metrics_html += """
        </tbody>
    </table>
    """

    # Add model formula if available
    model_html = ""
    if model_params and 'formula' in model_params and model_params.get('best_model', {}).get('r2', 0) > 0.1:
        r2 = model_params.get('best_model', {}).get('r2', 0)
        model_html = f"""
        <div class="metric">
            <div class="metric-label">Optimal Risk-Reward Model:</div>
            <div class="metric-value">{model_params['formula']} (R² = {r2:.2f})</div>
        </div>
        """

    # Create charts HTML
    charts_html = ""

    # Risk-Reward chart
    if 'risk_reward' in charts:
        embedded_img = embed_image(charts['risk_reward'])
        charts_html += f"""
        <div class="chart-container">
            <h3>Signal Confidence vs Optimal Risk-Reward Ratio</h3>
            <img src="{embedded_img}" alt="Confidence vs Risk-Reward">
            <div class="card-explanation">
                <p>This chart shows the optimal risk-reward ratio for different confidence levels based on historical performance. The trendline indicates the mathematical relationship between confidence and optimal risk-reward.</p>
            </div>
        </div>
        """

    # Win Rate chart
    if 'win_rate' in charts:
        embedded_img = embed_image(charts['win_rate'])
        charts_html += f"""
        <div class="chart-container">
            <h3>Signal Confidence vs Win Rate</h3>
            <img src="{embedded_img}" alt="Confidence vs Win Rate">
            <div class="card-explanation">
                <p>This chart shows how win rate changes with signal confidence. Ideally, higher confidence levels should correspond to higher win rates.</p>
            </div>
        </div>
        """

    # R-multiple chart
    if 'r_multiple' in charts:
        embedded_img = embed_image(charts['r_multiple'])
        charts_html += f"""
        <div class="chart-container">
            <h3>Signal Confidence vs R-Multiple</h3>
            <img src="{embedded_img}" alt="Confidence vs R-Multiple">
            <div class="card-explanation">
                <p>This chart shows the average R-multiple (profit/loss expressed in terms of initial risk) for each confidence level.</p>
            </div>
        </div>
        """

    # Trade scatter plot
    if 'trade_scatter' in charts:
        embedded_img = embed_image(charts['trade_scatter'])
        charts_html += f"""
        <div class="chart-container">
            <h3>Individual Trade Performance by Confidence</h3>
            <img src="{embedded_img}" alt="Trade Scatter Plot">
            <div class="card-explanation">
                <p>This scatter plot shows individual trade outcomes (R-multiple) by signal confidence level. Green points are winning trades, red points are losing trades.</p>
            </div>
        </div>
        """

    # Generate recommendations
    recommendations = []

    # Model-based recommendations
    if model_params and 'formula' in model_params and model_params.get('best_model', {}).get('r2', 0) > 0.3:
        recommendations.append(
            f"Implement the derived mathematical model ({model_params['formula']}) to dynamically adjust risk-reward ratios based on signal confidence.")

    # Performance-based recommendations
    if confidence_data:
        sorted_data = sorted(confidence_data, key=lambda x: x.get('confidence_group', 0))
        if len(sorted_data) >= 2:
            lowest_conf = sorted_data[0]
            highest_conf = sorted_data[-1]

            if highest_conf.get('win_mean', 0) > (lowest_conf.get('win_mean', 0) + 0.1):
                recommendations.append(
                    f"High confidence signals ({highest_conf.get('confidence_group', 0):.2f}) have significantly better win rates ({highest_conf.get('win_mean', 0):.2%}) than low confidence signals ({lowest_conf.get('confidence_group', 0):.2f}, {lowest_conf.get('win_mean', 0):.2%}).")

            if highest_conf.get('r_multiple_mean', 0) > (lowest_conf.get('r_multiple_mean', 0) + 0.3):
                recommendations.append(
                    f"High confidence signals yield better R-multiples ({highest_conf.get('r_multiple_mean', 0):.2f}) than low confidence signals ({lowest_conf.get('r_multiple_mean', 0):.2f}).")

    # Add general recommendation
    if confidence_data:
        best_conf = max(confidence_data, key=lambda x: x.get('r_multiple_mean', 0))
        best_conf_level = best_conf.get('confidence_group', 0)
        recommendations.append(
            f"The optimal performance is at confidence level {best_conf_level:.2f} with an average R-multiple of {best_conf.get('r_multiple_mean', 0):.2f}.")

    # Create recommendations HTML
    recommendations_html = ""
    if recommendations:
        recommendations_html = """
        <div class="recommendations">
            <h3>Confidence-Risk Analysis Insights</h3>
            <ul>
        """
        for rec in recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
            </ul>
        </div>
        """

    # Add explanation
    explanation_html = """
    <div class="card-explanation">
        <p><strong>What is the Confidence-Risk Analysis?</strong> This analysis examines the relationship between signal confidence and optimal risk-reward ratios based on historical performance.</p>
        <p><strong>Why is this important?</strong> It helps determine when to accept trades with different risk profiles based on the model's conviction level, potentially improving overall system performance.</p>
        <p><strong>How to use this?</strong> The mathematical relationship can be implemented in the risk management logic to dynamically adjust the required risk-reward ratio based on signal confidence.</p>
    </div>
    """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Confidence vs Risk-Reward Analysis</div>
        <div class="card-body">
            {explanation_html}

            {model_html}

            {charts_html}

            {metrics_html}

            {recommendations_html}
        </div>
    </div>
    """

    return html