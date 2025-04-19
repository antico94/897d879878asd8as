from ReportGeneration.ReportUtils.html_utils import get_badge_class, get_text_class


def generate_model_info_section(backtest_results):
    """Generate the model information section."""
    # Extract metadata for easy access
    model_info = backtest_results.get('model_info', {})
    metadata = model_info.get('metadata', {})

    # Get basic model information
    model_name = model_info.get('model_name', 'Gold Trading Model')
    timeframe = model_info.get('timeframe', metadata.get('timeframe', 'Unknown'))
    prediction_horizon = model_info.get('prediction_horizon', metadata.get('prediction_horizon', 1))
    prediction_target = metadata.get('prediction_target', 'direction')

    # Format timeframe for display
    timeframe_display = timeframe
    if timeframe == 'H1':
        timeframe_display = 'H1 (1 Hour)'
    elif timeframe == 'D1':
        timeframe_display = 'D1 (Daily)'
    elif timeframe == 'M15':
        timeframe_display = 'M15 (15 Minutes)'
    elif timeframe == 'M5':
        timeframe_display = 'M5 (5 Minutes)'

    # Convert prediction target for display
    target_display = "Price Direction (Up/Down)"
    if prediction_target == 'return':
        target_display = "Price Return (Percentage)"

    # Format data period
    data_period = model_info.get('data_period', {})
    start_date = data_period.get('start', 'Unknown')
    end_date = data_period.get('end', 'Unknown')

    # Get backtesting period
    backtest_start = backtest_results.get('start_date', start_date)
    backtest_end = backtest_results.get('end_date', end_date)

    # Get total trades information
    metrics = backtest_results.get('metrics', {})
    total_trades = metrics.get('total_trades', 0)

    # Get strategy parameters
    strategy_params = backtest_results.get('strategy_params', {})
    risk_per_trade = strategy_params.get('risk_per_trade', 0.02) * 100  # Convert to percentage

    # Create the metrics HTML
    metrics_html = f"""
    <div class="metric">
        <div class="metric-label">Model Name:</div>
        <div class="metric-value">{model_name}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Timeframe:</div>
        <div class="metric-value">{timeframe_display}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Prediction Target:</div>
        <div class="metric-value">{target_display}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Prediction Horizon:</div>
        <div class="metric-value">{prediction_horizon} {timeframe} period(s) ahead</div>
    </div>
    <div class="metric">
        <div class="metric-label">Backtesting Period:</div>
        <div class="metric-value">{backtest_start} - {backtest_end}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Total Trades:</div>
        <div class="metric-value">{total_trades}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Risk Per Trade:</div>
        <div class="metric-value">{risk_per_trade:.2f}% of account</div>
    </div>
    """

    # Add feature information if available
    if 'features' in metadata and metadata['features']:
        features = metadata['features']
        if isinstance(features, list):
            features_str = ", ".join(features)
        else:
            features_str = str(features)

        metrics_html += f"""
        <div class="metric">
            <div class="metric-label">Features Used:</div>
            <div class="metric-value">{len(features)} features</div>
        </div>
        <div class="card-explanation">
            <p><strong>Features:</strong> {features_str}</p>
        </div>
        """

    # Add strategy type if available
    strategy_type = backtest_results.get('strategy_type',
                                         strategy_params.get('strategy_type', 'ML-based Trading Strategy'))
    metrics_html += f"""
    <div class="metric">
        <div class="metric-label">Strategy Type:</div>
        <div class="metric-value">{strategy_type}</div>
    </div>
    """

    # Add any additional strategy-specific parameters
    if strategy_params:
        additional_params = []

        # Position sizing approach
        position_size_mode = strategy_params.get('position_size_mode', 'risk')
        if position_size_mode == 'risk':
            additional_params.append(f"Risk-based position sizing ({risk_per_trade:.2f}% per trade)")
        elif position_size_mode == 'fixed':
            fixed_size = strategy_params.get('fixed_position_size', 0.1)
            additional_params.append(f"Fixed position sizing ({fixed_size} lots per trade)")

        # Stop loss / take profit
        use_sl_tp = strategy_params.get('use_sl_tp', True)
        if use_sl_tp:
            additional_params.append("Using stop loss and take profit levels")

        # Partial closing
        enable_partial = strategy_params.get('enable_partial_close', False)
        if enable_partial:
            additional_params.append("Partial position closing enabled")

        # Breakeven
        enable_breakeven = strategy_params.get('enable_breakeven', False)
        if enable_breakeven:
            additional_params.append("Move to breakeven enabled")

        # Display additional parameters
        if additional_params:
            metrics_html += """
            <div class="card-explanation">
                <p><strong>Strategy Parameters:</strong></p>
                <ul>
            """

            for param in additional_params:
                metrics_html += f"<li>{param}</li>"

            metrics_html += """
                </ul>
            </div>
            """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Model and Strategy Information</div>
        <div class="card-body">
            {metrics_html}

            <div class="card-explanation">
                <p><strong>What is this strategy?</strong> This strategy uses machine learning to predict gold price movements and trades accordingly. It was backtested on historical gold price data to evaluate its performance.</p>
            </div>
        </div>
    </div>
    """

    return html