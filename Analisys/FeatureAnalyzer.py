import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import StandardScaler
import os

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.DataStorage import DataStorage


class FeatureAnalyzer:
    """Analyzes features to identify the most important indicators."""

    def __init__(self, config: Config, logger: Logger, data_storage: DataStorage):
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.output_dir = "FeatureAnalysis"
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, pair: str = "XAUUSD", timeframe: str = "H1",
                  dataset_type: str = "training") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load processed data for analysis."""
        try:
            self.logger.info(f"Loading {dataset_type} data for {pair} {timeframe}")
            X, y = self.data_storage.load_processed_data(pair, timeframe, dataset_type)

            if X.empty:
                self.logger.warning(f"No data found for {pair} {timeframe} {dataset_type}")
                return pd.DataFrame(), pd.DataFrame()

            # Remove 'time' column as it's not a feature for ML
            if 'time' in X.columns:
                self.time_data = X['time'].copy()
                X = X.drop('time', axis=1)

            self.logger.info(f"Loaded {len(X)} rows with {len(X.columns)} features")
            return X, y

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def analyze_correlation(self, X: pd.DataFrame, threshold: float = 0.85) -> Dict[str, List[str]]:
        """Identify highly correlated features."""
        try:
            self.logger.info(f"Analyzing feature correlations with threshold {threshold}")

            # Calculate correlation matrix
            corr_matrix = X.corr().abs()

            # Create a dictionary to store correlated groups
            correlated_features = {}

            # Find highly correlated feature pairs
            for i in range(len(corr_matrix.columns)):
                col_name = corr_matrix.columns[i]
                # Get correlations for the current column that are above threshold
                # and only look at the lower triangle to avoid duplicates
                corr_values = corr_matrix.iloc[i + 1:, i]
                high_corr = corr_values[corr_values > threshold]

                if not high_corr.empty:
                    correlated_cols = high_corr.index.tolist()
                    correlated_features[col_name] = correlated_cols
                    self.logger.info(
                        f"Feature {col_name} is highly correlated with {len(correlated_cols)} other features")

            # Create correlation heatmap
            plt.figure(figsize=(20, 16))
            mask = np.triu(corr_matrix)
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm',
                        annot=False, square=True, linewidths=.5)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/correlation_heatmap.png")
            plt.close()

            return correlated_features

        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            raise

    def analyze_feature_importance(self, X: pd.DataFrame, y: pd.DataFrame,
                                   target_col: str = 'future_price_1') -> pd.DataFrame:
        """Analyze feature importance using Random Forest."""
        try:
            if target_col not in y.columns:
                self.logger.error(f"Target column {target_col} not found in target data")
                return pd.DataFrame()

            self.logger.info(f"Analyzing feature importance for target {target_col}")

            # Scale features for better model performance
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Use Random Forest for feature importance
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_scaled, y[target_col])

            # Get feature importances
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Plot feature importances
            plt.figure(figsize=(12, 10))
            top_features = feature_importance.head(30)
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title(f'Top 30 Feature Importances for {target_col}')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance_{target_col}.png")
            plt.close()

            self.logger.info(
                f"Top 5 most important features: {', '.join(feature_importance['Feature'].head(5).tolist())}")
            return feature_importance

        except Exception as e:
            self.logger.error(f"Error analyzing feature importance: {e}")
            raise

    def select_features_rfe(self, X: pd.DataFrame, y: pd.DataFrame,
                            target_col: str = 'future_price_1', n_features: int = 30) -> List[str]:
        """Select optimal features using Recursive Feature Elimination."""
        try:
            if target_col not in y.columns:
                self.logger.error(f"Target column {target_col} not found in target data")
                return []

            self.logger.info(f"Performing RFE to select top {n_features} features for {target_col}")

            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Use Random Forest with RFE
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            rfe = RFE(estimator=model, n_features_to_select=n_features, step=5)
            rfe.fit(X_scaled, y[target_col])

            # Get selected features
            selected_features = X.columns[rfe.support_].tolist()

            self.logger.info(f"Selected {len(selected_features)} features through RFE")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in RFE feature selection: {e}")
            raise

    def select_features_statistical(self, X: pd.DataFrame, y: pd.DataFrame,
                                    target_col: str = 'future_price_1', k: int = 30) -> List[str]:
        """Select features using statistical tests (F-regression)."""
        try:
            if target_col not in y.columns:
                self.logger.error(f"Target column {target_col} not found in target data")
                return []

            self.logger.info(f"Selecting top {k} features using F-regression for {target_col}")

            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Apply SelectKBest with f_regression
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X_scaled, y[target_col])

            # Get mask of selected features
            selected_features = X.columns[selector.get_support()].tolist()

            self.logger.info(f"Selected {len(selected_features)} features through statistical testing")
            return selected_features

        except Exception as e:
            self.logger.error(f"Error in statistical feature selection: {e}")
            raise

    def remove_redundant_features(self, X: pd.DataFrame, correlated_features: Dict[str, List[str]],
                                  importance_df: pd.DataFrame) -> List[str]:
        """Remove redundant features based on correlation and importance."""
        try:
            self.logger.info("Removing redundant features")

            # Start with all features
            features_to_keep = X.columns.tolist()

            # For each group of correlated features, keep only the most important one
            for feature, correlated in correlated_features.items():
                if feature not in features_to_keep:
                    continue

                # Add the feature itself to the list of correlated features
                group = [feature] + correlated

                # Find which feature in the group has the highest importance
                group_importances = importance_df[importance_df['Feature'].isin(group)]
                if not group_importances.empty:
                    most_important = group_importances.iloc[0]['Feature']

                    # Remove all other features in this correlated group
                    for feat in group:
                        if feat != most_important and feat in features_to_keep:
                            features_to_keep.remove(feat)
                            self.logger.debug(f"Removed {feat} as redundant with {most_important}")

            self.logger.info(f"Kept {len(features_to_keep)} features after removing redundancies")
            return features_to_keep

        except Exception as e:
            self.logger.error(f"Error removing redundant features: {e}")
            raise

    def generate_feature_report(self, importance_df: pd.DataFrame, selected_features: List[str],
                                correlated_groups: Dict[str, List[str]]) -> None:
        """Generate a summary report of feature analysis."""
        try:
            report_path = f"{self.output_dir}/feature_analysis_report.txt"

            with open(report_path, 'w') as f:
                f.write("FEATURE ANALYSIS REPORT\n")
                f.write("======================\n\n")

                # Write top features by importance
                f.write("TOP 20 FEATURES BY IMPORTANCE:\n")
                f.write("-----------------------------\n")
                for idx, row in importance_df.head(20).iterrows():
                    f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

                f.write("\n\n")

                # Write selected features after redundancy removal
                f.write("SELECTED FEATURES AFTER REDUNDANCY REMOVAL:\n")
                f.write("-----------------------------------------\n")
                for feature in selected_features:
                    f.write(f"{feature}\n")

                f.write("\n\n")

                # Write correlation groups
                f.write("HIGHLY CORRELATED FEATURE GROUPS:\n")
                f.write("--------------------------------\n")
                for feature, correlated in correlated_groups.items():
                    if correlated:
                        f.write(f"{feature} correlated with: {', '.join(correlated)}\n")

            self.logger.info(f"Feature analysis report saved to {report_path}")

        except Exception as e:
            self.logger.error(f"Error generating feature report: {e}")
            raise

    def run_complete_analysis(self, pair: str = "XAUUSD", timeframe: str = "H1",
                              dataset_type: str = "training",
                              target_col: str = "future_price_1") -> List[str]:
        """Run complete feature analysis pipeline."""
        try:
            # Load the data
            X, y = self.load_data(pair, timeframe, dataset_type)
            if X.empty or y.empty:
                return []

            # Analyze correlations
            correlated_features = self.analyze_correlation(X, threshold=0.85)

            # Get feature importance
            importance_df = self.analyze_feature_importance(X, y, target_col)

            # Select features with RFE
            rfe_features = self.select_features_rfe(X, y, target_col)

            # Select features with statistical tests
            stat_features = self.select_features_statistical(X, y, target_col)

            # Find common features between the two methods
            common_features = list(set(rfe_features).intersection(set(stat_features)))
            self.logger.info(f"Found {len(common_features)} common features between RFE and statistical selection")

            # Remove redundant features
            final_features = self.remove_redundant_features(
                X[common_features],
                {k: [f for f in v if f in common_features] for k, v in correlated_features.items() if
                 k in common_features},
                importance_df
            )

            # Generate report
            self.generate_feature_report(importance_df, final_features, correlated_features)

            return final_features

        except Exception as e:
            self.logger.error(f"Error in complete feature analysis: {e}")
            raise