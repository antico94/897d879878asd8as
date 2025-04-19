import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set matplotlib backend to non-interactive
matplotlib.use('Agg')
# Set style
plt.style.use('seaborn-v0_8-darkgrid')


def create_output_dir():
    """Create and return the output directory for charts."""
    output_dir = Path("BacktestResults") / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_equity_curve_chart(equity_data, initial_balance=10000):
    """Generate equity curve chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data
    if isinstance(equity_data, dict):
        # Handle dict format
        timestamps = [point.get('timestamp') for point in equity_data.get('equity_curve', [])]
        equity = [point.get('equity') for point in equity_data.get('equity_curve', [])]
        balance = [point.get('balance') for point in equity_data.get('equity_curve', [])]
    else:
        # Handle DataFrame format
        timestamps = equity_data.index
        if 'equity' in equity_data.columns:
            equity = equity_data['equity'].values
        else:
            equity = equity_data['balance'].values

        if 'balance' in equity_data.columns:
            balance = equity_data['balance'].values
        else:
            balance = equity

    # Create the plot
    plt.figure(figsize=(12, 6))

    if len(timestamps) > 0:
        # Plot equity curve
        plt.plot(timestamps, equity, label='Equity', color='#3498db', linewidth=2)

        # Plot balance if different from equity
        if not np.array_equal(equity, balance):
            plt.plot(timestamps, balance, label='Balance', color='#2ecc71', linewidth=1.5, linestyle='--')

        # Add initial balance reference line
        plt.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.5,
                    label=f'Initial Balance (${initial_balance:,.2f})')

    plt.title('Equity Curve', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Account Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve date formatting for x-axis if timestamps are datetime
    if len(timestamps) > 0 and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.gcf().autofmt_xdate()

    # Save the chart
    output_path = output_dir / f"equity_curve_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_drawdown_chart(equity_data):
    """Generate drawdown chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data
    if isinstance(equity_data, dict):
        # Handle dict format
        timestamps = [point.get('timestamp') for point in equity_data.get('equity_curve', [])]
        equity = np.array([point.get('equity') for point in equity_data.get('equity_curve', [])])
    else:
        # Handle DataFrame format
        timestamps = equity_data.index
        if 'equity' in equity_data.columns:
            equity = equity_data['equity'].values
        else:
            equity = equity_data['balance'].values

    # Calculate drawdown
    if len(equity) > 0:
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100  # Convert to percentage

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, drawdown, color='#e74c3c', linewidth=2)
        plt.fill_between(timestamps, drawdown, 0, color='#e74c3c', alpha=0.3)

        plt.title('Drawdown (%)', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Improve date formatting for x-axis if timestamps are datetime
        if isinstance(timestamps[0], (datetime, pd.Timestamp)):
            plt.gcf().autofmt_xdate()

        # Invert y-axis to show drawdowns as negative values
        plt.gca().invert_yaxis()

        # Save the chart
        output_path = output_dir / f"drawdown_{timestamp}.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return str(output_path)

    return None


def generate_monthly_returns_chart(returns_data):
    """Generate monthly returns chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if isinstance(returns_data, dict):
        # Convert dict to Series or similar format
        months = list(returns_data.keys())
        returns = list(returns_data.values())
    elif isinstance(returns_data, pd.Series):
        months = returns_data.index
        returns = returns_data.values
    else:
        # Assume it's a DataFrame with date index and 'return' column
        months = returns_data.index
        returns = returns_data['return'].values

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Create colors for bars (green for positive, red for negative)
    colors = ['#2ecc71' if ret >= 0 else '#e74c3c' for ret in returns]

    # Create bar plot
    bars = plt.bar(months, returns, color=colors, alpha=0.7)

    # Add horizontal line at 0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    plt.title('Monthly Returns (%)', fontsize=14)
    plt.ylabel('Return (%)', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)

    # Improve date formatting for x-axis
    plt.xticks(rotation=45)

    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height + (0.1 if height >= 0 else -0.5),
            f'{height:.1f}%',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=9
        )

    plt.tight_layout()

    # Save the chart
    output_path = output_dir / f"monthly_returns_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_win_loss_chart(trades_data):
    """Generate win/loss distribution chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract profit/loss data
    if isinstance(trades_data, list):
        profits = [trade.get('profit_loss', 0) for trade in trades_data]
    elif isinstance(trades_data, pd.DataFrame):
        profits = trades_data['profit_loss'].values
    else:
        profits = []

    if not profits:
        return None

    plt.figure(figsize=(12, 6))

    # Create histogram with KDE
    sns.histplot(profits, bins=20, kde=True, color='#3498db')

    # Add vertical line at 0
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)

    # Add labels
    plt.title('Profit/Loss Distribution', fontsize=14)
    plt.xlabel('Profit/Loss ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save the chart
    output_path = output_dir / f"profit_loss_distribution_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    # Create cumulative P&L chart
    plt.figure(figsize=(12, 6))

    # Calculate cumulative P&L
    cumulative_pnl = np.cumsum(profits)

    # Plot cumulative P&L
    plt.plot(cumulative_pnl, color='#2ecc71', linewidth=2)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Add labels
    plt.title('Cumulative Profit/Loss', fontsize=14)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative Profit/Loss ($)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save the chart
    cumulative_path = output_dir / f"cumulative_pnl_{timestamp}.png"
    plt.savefig(cumulative_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path), str(cumulative_path)


def generate_trade_duration_chart(trades_data):
    """Generate trade duration distribution chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract duration data
    if isinstance(trades_data, list):
        durations = []
        for trade in trades_data:
            if 'trade_duration' in trade:
                durations.append(trade['trade_duration'])
            elif 'entry_time' in trade and 'exit_time' in trade:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                duration_hours = (exit_time - entry_time).total_seconds() / 3600
                durations.append(duration_hours)
    elif isinstance(trades_data, pd.DataFrame):
        if 'trade_duration' in trades_data.columns:
            durations = trades_data['trade_duration'].values
        elif 'entry_time' in trades_data.columns and 'exit_time' in trades_data.columns:
            durations = (pd.to_datetime(trades_data['exit_time']) -
                         pd.to_datetime(trades_data['entry_time'])).dt.total_seconds() / 3600
        else:
            durations = []
    else:
        durations = []

    if not durations:
        return None

    plt.figure(figsize=(12, 6))

    # Create histogram with KDE
    sns.histplot(durations, bins=20, kde=True, color='#9b59b6')

    # Add labels
    plt.title('Trade Duration Distribution', fontsize=14)
    plt.xlabel('Duration (Hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save the chart
    output_path = output_dir / f"trade_duration_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_trade_size_chart(trades_data):
    """Generate trade size distribution chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract position size data
    if isinstance(trades_data, list):
        sizes = [trade.get('position_size', 0) for trade in trades_data]
    elif isinstance(trades_data, pd.DataFrame):
        sizes = trades_data['position_size'].values
    else:
        sizes = []

    if not sizes or all(s == 0 for s in sizes):
        return None

    plt.figure(figsize=(12, 6))

    # Create histogram
    plt.hist(sizes, bins=15, alpha=0.7, color='#f39c12')

    # Add labels
    plt.title('Position Size Distribution', fontsize=14)
    plt.xlabel('Position Size (Lots)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save the chart
    output_path = output_dir / f"position_size_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_win_rate_by_day_chart(trades_data):
    """Generate win rate by day of week chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process trades into DataFrame if needed
    if isinstance(trades_data, list):
        trades_df = pd.DataFrame(trades_data)
    elif isinstance(trades_data, pd.DataFrame):
        trades_df = trades_data
    else:
        return None

    # Ensure we have entry time and win information
    if 'entry_time' not in trades_df.columns or (
            'win' not in trades_df.columns and 'profit_loss' not in trades_df.columns):
        return None

    # Convert entry_time to datetime if it's not already
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    # Create 'win' column if it doesn't exist
    if 'win' not in trades_df.columns and 'profit_loss' in trades_df.columns:
        trades_df['win'] = trades_df['profit_loss'] > 0

    # Extract day of week
    trades_df['day_of_week'] = trades_df['entry_time'].dt.day_name()

    # Ensure days are in correct order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Group by day of week and calculate win rate
    win_rate_by_day = trades_df.groupby('day_of_week')['win'].mean().reindex(
        [d for d in day_order if d in trades_df['day_of_week'].unique()]
    )

    # Also get trade counts by day
    count_by_day = trades_df.groupby('day_of_week').size().reindex(win_rate_by_day.index)

    plt.figure(figsize=(12, 6))

    # Create bar chart for win rate
    ax = win_rate_by_day.plot(kind='bar', color='#3498db', alpha=0.7)

    # Add horizontal line at 0.5 (50% win rate)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Add labels above bars
    for i, v in enumerate(win_rate_by_day):
        count = count_by_day[i]
        ax.text(i, v + 0.02, f"{v:.1%} ({count} trades)", ha='center', fontsize=10)

    plt.title('Win Rate by Day of Week', fontsize=14)
    plt.xlabel('Day of Week', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)

    # Save the chart
    output_path = output_dir / f"win_rate_by_day_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_win_rate_by_hour_chart(trades_data):
    """Generate win rate by hour chart and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Process trades into DataFrame if needed
    if isinstance(trades_data, list):
        trades_df = pd.DataFrame(trades_data)
    elif isinstance(trades_data, pd.DataFrame):
        trades_df = trades_data
    else:
        return None

    # Ensure we have entry time and win information
    if 'entry_time' not in trades_df.columns or (
            'win' not in trades_df.columns and 'profit_loss' not in trades_df.columns):
        return None

    # Convert entry_time to datetime if it's not already
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    # Create 'win' column if it doesn't exist
    if 'win' not in trades_df.columns and 'profit_loss' in trades_df.columns:
        trades_df['win'] = trades_df['profit_loss'] > 0

    # Extract hour of day
    trades_df['hour'] = trades_df['entry_time'].dt.hour

    # Group by hour and calculate win rate
    win_rate_by_hour = trades_df.groupby('hour')['win'].mean()

    # Also get trade counts by hour
    count_by_hour = trades_df.groupby('hour').size()

    plt.figure(figsize=(14, 6))

    # Create bar chart for win rate
    ax = win_rate_by_hour.plot(kind='bar', color='#2ecc71', alpha=0.7)

    # Add horizontal line at 0.5 (50% win rate)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

    # Add labels above bars
    for i, v in enumerate(win_rate_by_hour):
        # Fix: Make sure the index exists in count_by_hour
        hour_idx = win_rate_by_hour.index[i]
        count = count_by_hour.get(hour_idx, 0)  # Use .get() with default value
        ax.text(i, v + 0.02, f"{v:.1%} ({count})", ha='center', fontsize=9)

    plt.title('Win Rate by Hour of Day', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, axis='y', alpha=0.3)

    # Ensure all hours are labeled
    plt.xticks(range(len(win_rate_by_hour)), [f"{h:02d}:00" for h in win_rate_by_hour.index])

    # Save the chart
    output_path = output_dir / f"win_rate_by_hour_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)


def generate_balance_comparison_chart(backtest_results, baseline_returns=None):
    """Generate chart comparing strategy to baseline (e.g., buy & hold) and return the path."""
    output_dir = create_output_dir()

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract equity curve data
    if isinstance(backtest_results, dict) and 'equity_curve' in backtest_results:
        timestamps = [point.get('timestamp') for point in backtest_results['equity_curve']]
        equity = [point.get('equity') for point in backtest_results['equity_curve']]
        initial_balance = backtest_results.get('initial_balance', 10000)
    elif isinstance(backtest_results, pd.DataFrame):
        timestamps = backtest_results.index
        if 'equity' in backtest_results.columns:
            equity = backtest_results['equity'].values
        else:
            equity = backtest_results['balance'].values
        initial_balance = equity[0] if len(equity) > 0 else 10000
    else:
        return None

    plt.figure(figsize=(12, 6))

    # Plot strategy equity curve
    plt.plot(timestamps, equity, label='Strategy', color='#3498db', linewidth=2)

    # Add baseline if provided
    if baseline_returns is not None:
        if isinstance(baseline_returns, pd.Series):
            # Convert returns to equity curve
            baseline_equity = initial_balance * (1 + baseline_returns).cumprod()
            plt.plot(baseline_returns.index, baseline_equity,
                     label='Buy & Hold', color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)
        elif isinstance(baseline_returns, pd.DataFrame):
            # Assume first column is returns
            baseline_equity = initial_balance * (1 + baseline_returns.iloc[:, 0]).cumprod()
            plt.plot(baseline_returns.index, baseline_equity,
                     label='Buy & Hold', color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7)

    # Add initial balance reference line
    plt.axhline(y=initial_balance, color='#34495e', linestyle=':', alpha=0.7,
                label=f'Initial Balance (${initial_balance:,.2f})')

    plt.title('Strategy vs. Baseline Performance', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Account Value ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve date formatting for x-axis if timestamps are datetime
    if len(timestamps) > 0 and isinstance(timestamps[0], (datetime, pd.Timestamp)):
        plt.gcf().autofmt_xdate()

    # Save the chart
    output_path = output_dir / f"strategy_comparison_{timestamp}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return str(output_path)