import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

class ConfidenceRiskAnalyzer:
    """Analyzes the relationship between signal confidence and optimal risk-reward ratios."""

    def __init__(self, logger):
        self.logger = logger

    def analyze_backtest_data(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze backtest trades to find optimal confidence-risk relationship.

        Args:
            trades: List of trade dictionaries from backtest results

        Returns:
            Dictionary with analysis results and model parameters
        """
        try:
            if not trades:
                self.logger.warning("No trades available for confidence-risk analysis")
                return {}

            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(trades)

            # Ensure required columns exist
            required_cols = ['signal_strength', 'profit_loss', 'entry_price', 'stop_loss', 'take_profit']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for analysis. Required: {required_cols}")
                return {}

            # Calculate actual risk-reward ratio for each trade
            df['risk'] = abs(df['entry_price'] - df['stop_loss'])
            df['reward'] = abs(df['entry_price'] - df['take_profit'])
            df['risk_reward_ratio'] = df['reward'] / df['risk']

            # Calculate R-multiple (actual profit/loss divided by risk)
            df['r_multiple'] = df['profit_loss'] / (df['risk'] * 10 * df['position_size'])

            # Group by confidence levels (rounded to 0.05 intervals for better statistics)
            df['confidence_group'] = (df['signal_strength'] * 20).round() / 20

            # Calculate key metrics per confidence group
            confidence_analysis = df.groupby('confidence_group').agg({
                'profit_loss': ['mean', 'count', 'sum'],
                'r_multiple': ['mean', 'median', 'std'],
                'risk_reward_ratio': ['mean', 'median'],
                'win': ['mean']
            }).reset_index()

            # Flatten multi-level columns
            confidence_analysis.columns = ['_'.join(col).strip('_') for col in confidence_analysis.columns.values]

            # Find optimal risk-reward ratio for each confidence level
            # (using the ratio that would have maximized returns)
            optimal_rr = []
            for conf_level in confidence_analysis['confidence_group']:
                group_trades = df[df['confidence_group'] == conf_level]
                best_rr, best_pnl = self._find_optimal_risk_reward(group_trades)
                optimal_rr.append({
                    'confidence': conf_level,
                    'optimal_risk_reward': best_rr,
                    'projected_pnl': best_pnl
                })

            optimal_rr_df = pd.DataFrame(optimal_rr)

            # Merge optimal RR with analysis
            results_df = pd.merge(confidence_analysis, optimal_rr_df,
                                 left_on='confidence_group',
                                 right_on='confidence')

            # Fit mathematical model to describe confidence vs optimal risk-reward
            model_params = self._fit_confidence_risk_model(results_df)

            # Generate visualizations
            charts = self._generate_analysis_charts(results_df, model_params, df)

            # Return complete analysis
            return {
                'confidence_analysis': results_df.to_dict('records'),
                'model_params': model_params,
                'charts': charts,
                'raw_data': df
            }

        except Exception as e:
            self.logger.error(f"Error analyzing confidence-risk relationship: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def _find_optimal_risk_reward(self, trades_df: pd.DataFrame) -> Tuple[float, float]:
        """Find the risk-reward ratio that would have maximized returns for a set of trades."""
        if trades_df.empty:
            return 1.5, 0  # Default fallback

        # Test different risk-reward ratios
        test_ratios = np.arange(1.0, 3.1, 0.1)
        results = []

        for ratio in test_ratios:
            # Simulate trades with this fixed risk-reward ratio
            pnl = self._simulate_fixed_risk_reward(trades_df, ratio)
            results.append((ratio, pnl))

        # Find ratio with maximum PnL
        optimal_result = max(results, key=lambda x: x[1])
        return optimal_result

    def _simulate_fixed_risk_reward(self, trades_df: pd.DataFrame, risk_reward: float) -> float:
        """Simulate trades using a fixed risk-reward ratio."""
        total_pnl = 0

        for _, trade in trades_df.iterrows():
            risk = abs(trade['entry_price'] - trade['stop_loss'])
            direction = 1 if trade['direction'].endswith('BUY') else -1

            # Calculate take profit based on risk-reward
            simulated_tp = trade['entry_price'] + (direction * risk * risk_reward)

            # See if take profit would have been hit before stop loss
            if trade['win']:
                # Winner in reality - would it win with this TP?
                if direction == 1:  # BUY
                    if trade['exit_price'] >= simulated_tp:
                        # TP hit
                        gain = risk * risk_reward
                    else:
                        # Exited earlier
                        gain = trade['exit_price'] - trade['entry_price']
                else:  # SELL
                    if trade['exit_price'] <= simulated_tp:
                        # TP hit
                        gain = risk * risk_reward
                    else:
                        # Exited earlier
                        gain = trade['entry_price'] - trade['exit_price']

                total_pnl += gain * 10 * trade['position_size']  # Convert to dollars
            else:
                # Loser in reality - would have lost full risk
                total_pnl -= risk * 10 * trade['position_size']  # Convert to dollars

        return total_pnl

    def _fit_confidence_risk_model(self, analysis_df: pd.DataFrame) -> Dict[str, Any]:
        """Fit mathematical model to the confidence vs optimal risk-reward relationship."""
        if analysis_df.empty or 'confidence_group' not in analysis_df.columns:
            return {}

        # Extract relevant data
        x = analysis_df['confidence_group'].values
        y = analysis_df['optimal_risk_reward'].values

        # Try different models to find best fit
        models = {
            'linear': lambda x, y: stats.linregress(x, y),
            'logarithmic': lambda x, y: stats.linregress(np.log(x), y) if all(x > 0) else None,
            'exponential': lambda x, y: stats.linregress(x, np.log(y)) if all(y > 0) else None,
            'power': lambda x, y: stats.linregress(np.log(x), np.log(y)) if all(x > 0) and all(y > 0) else None
        }

        results = {}
        best_model = {'name': 'linear', 'r2': 0}

        for name, model_fn in models.items():
            try:
                fit_result = model_fn(x, y)
                if fit_result is not None:
                    # Calculate R² to find best fit
                    if name == 'linear':
                        y_pred = fit_result.slope * x + fit_result.intercept
                        r2 = r2_score(y, y_pred)
                    elif name == 'logarithmic':
                        y_pred = fit_result.slope * np.log(x) + fit_result.intercept
                        r2 = r2_score(y, y_pred)
                    elif name == 'exponential':
                        y_pred = np.exp(fit_result.intercept) * np.exp(fit_result.slope * x)
                        r2 = r2_score(y, np.log(y_pred))
                    elif name == 'power':
                        y_pred = np.exp(fit_result.intercept) * x ** fit_result.slope
                        r2 = r2_score(np.log(y), np.log(y_pred))

                    results[name] = {
                        'params': {k: getattr(fit_result, k) for k in ['slope', 'intercept', 'rvalue', 'pvalue', 'stderr']},
                        'r2': r2
                    }

                    if r2 > best_model['r2']:
                        best_model = {'name': name, 'r2': r2}
            except Exception as e:
                self.logger.warning(f"Error fitting {name} model: {e}")

        return {
            'models': results,
            'best_model': best_model,
            'formula': self._get_model_formula(best_model['name'],
                                              results.get(best_model['name'], {}).get('params', {}))
        }

    def _get_model_formula(self, model_name: str, params: Dict[str, float]) -> str:
        """Get the formula for the selected model."""
        if not params:
            return "No valid model found"

        slope = params.get('slope', 0)
        intercept = params.get('intercept', 0)

        if model_name == 'linear':
            return f"RR = {slope:.4f} * confidence + {intercept:.4f}"
        elif model_name == 'logarithmic':
            return f"RR = {slope:.4f} * ln(confidence) + {intercept:.4f}"
        elif model_name == 'exponential':
            return f"RR = {np.exp(intercept):.4f} * e^({slope:.4f} * confidence)"
        elif model_name == 'power':
            return f"RR = {np.exp(intercept):.4f} * confidence^{slope:.4f}"
        else:
            return "Unknown model type"

    def _generate_analysis_charts(self, analysis_df: pd.DataFrame,
                                 model_params: Dict[str, Any],
                                 trades_df: pd.DataFrame) -> Dict[str, str]:
        """Generate charts for the confidence-risk analysis."""
        output_dir = self._create_output_dir()
        charts = {}

        # 1. Plot confidence vs win rate
        if not analysis_df.empty and 'confidence_group' in analysis_df.columns and 'win_mean' in analysis_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(analysis_df['confidence_group'], analysis_df['win_mean'], s=analysis_df['profit_loss_count']*5)
            plt.plot(analysis_df['confidence_group'], analysis_df['win_mean'], 'b-')
            plt.title('Signal Confidence vs Win Rate')
            plt.xlabel('Signal Confidence')
            plt.ylabel('Win Rate')
            plt.grid(alpha=0.3)

            # Add sample sizes
            for _, row in analysis_df.iterrows():
                plt.annotate(f"n={int(row['profit_loss_count'])}",
                             (row['confidence_group'], row['win_mean']),
                             textcoords="offset points",
                             xytext=(0,10),
                             ha='center')

            # Save chart
            chart_path = os.path.join(output_dir, 'confidence_vs_winrate.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            charts['win_rate'] = chart_path

        # 2. Plot confidence vs optimal risk-reward
        if not analysis_df.empty and 'confidence_group' in analysis_df.columns and 'optimal_risk_reward' in analysis_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(analysis_df['confidence_group'], analysis_df['optimal_risk_reward'],
                      s=analysis_df['profit_loss_count']*5)

            # Plot best-fit line using the optimal model
            if model_params and 'best_model' in model_params and 'models' in model_params:
                best_model = model_params['best_model']['name']
                if best_model in model_params['models']:
                    params = model_params['models'][best_model]['params']
                    x = np.linspace(min(analysis_df['confidence_group']), max(analysis_df['confidence_group']), 100)

                    if best_model == 'linear':
                        y = params['slope'] * x + params['intercept']
                    elif best_model == 'logarithmic':
                        y = params['slope'] * np.log(x) + params['intercept']
                    elif best_model == 'exponential':
                        y = np.exp(params['intercept']) * np.exp(params['slope'] * x)
                    elif best_model == 'power':
                        y = np.exp(params['intercept']) * x ** params['slope']

                    plt.plot(x, y, 'r-', label=f'Model: {model_params["formula"]}')
                    plt.legend()

            plt.title('Signal Confidence vs Optimal Risk-Reward Ratio')
            plt.xlabel('Signal Confidence')
            plt.ylabel('Optimal Risk-Reward Ratio')
            plt.grid(alpha=0.3)

            # Add min and max guidelines
            plt.axhline(y=1.5, color='gray', linestyle='--', alpha=0.7, label='Minimum 1.5:1')
            plt.axhline(y=2.0, color='gray', linestyle=':', alpha=0.7, label='Standard 2:1')

            # Save chart
            chart_path = os.path.join(output_dir, 'confidence_vs_risk_reward.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            charts['risk_reward'] = chart_path

        # 3. Plot confidence vs average R-multiple
        if not analysis_df.empty and 'confidence_group' in analysis_df.columns and 'r_multiple_mean' in analysis_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(analysis_df['confidence_group'], analysis_df['r_multiple_mean'],
                      s=analysis_df['profit_loss_count']*5)
            plt.plot(analysis_df['confidence_group'], analysis_df['r_multiple_mean'], 'g-')
            plt.title('Signal Confidence vs Average R-Multiple')
            plt.xlabel('Signal Confidence')
            plt.ylabel('Average R-Multiple')
            plt.grid(alpha=0.3)

            # Add reference line at R=0
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)

            # Save chart
            chart_path = os.path.join(output_dir, 'confidence_vs_r_multiple.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            charts['r_multiple'] = chart_path

        # 4. Scatter plot of all trades by confidence and R-multiple
        if not trades_df.empty and 'signal_strength' in trades_df.columns and 'r_multiple' in trades_df.columns:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(trades_df['signal_strength'],
                                trades_df['r_multiple'],
                                c=trades_df['win'].map({True: 'green', False: 'red'}),
                                alpha=0.6)

            plt.title('Trade Performance by Signal Confidence')
            plt.xlabel('Signal Confidence')
            plt.ylabel('R-Multiple (Profit/Loss in terms of risk)')
            plt.grid(alpha=0.3)

            # Add reference line at R=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Add legend
            plt.legend(*scatter.legend_elements(), title="Outcome", loc="upper left")

            # Save chart
            chart_path = os.path.join(output_dir, 'trade_scatter_by_confidence.png')
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            charts['trade_scatter'] = chart_path

        return charts

    def _create_output_dir(self) -> str:
        """Create and return the output directory for charts."""
        output_dir = os.path.join("BacktestResults", "confidence_analysis")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir