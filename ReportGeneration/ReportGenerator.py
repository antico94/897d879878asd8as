import os
import webbrowser
import logging
from pathlib import Path
from datetime import datetime

from ReportGeneration.Sections.model_info_section import generate_model_info_section
from ReportGeneration.Sections.accuracy_section import generate_accuracy_section
from ReportGeneration.Sections.trades_section import generate_trades_section
from ReportGeneration.Sections.equity_section import generate_equity_section
from ReportGeneration.Sections.performance_section import generate_performance_section
from ReportGeneration.Sections.summary_section import generate_summary_section
from ReportGeneration.ReportUtils.html_utils import create_html_report


def generate_backtest_report(backtest_results, output_dir=None, open_browser=True):
    logger = logging.getLogger("BacktestReport")

    try:
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = Path("BacktestResults")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for report naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"backtest_report_{timestamp}.html"
        report_path = output_dir / report_name

        logger.info(f"Generating backtest report at {report_path}")

        # Generate Sections
        model_info_html = generate_model_info_section(backtest_results)
        accuracy_html = generate_accuracy_section(backtest_results)
        trades_html = generate_trades_section(backtest_results)
        equity_html = generate_equity_section(backtest_results)
        performance_html = generate_performance_section(backtest_results)
        summary_html = generate_summary_section(backtest_results)

        # Create the full HTML report
        html_content = create_html_report({
            'model_info': model_info_html,
            'summary': summary_html,
            'performance': performance_html,
            'equity': equity_html,
            'trades': trades_html,
            'accuracy': accuracy_html
        }, timestamp)

        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"Backtest report successfully generated at {report_path}")

        # Open the report in the default browser
        if open_browser:
            # Use absolute path for browser opening
            abs_report_path = os.path.abspath(report_path)
            try:
                logger.info(f"Opening report in browser: {abs_report_path}")
                webbrowser.open(f"file://{abs_report_path}")
            except Exception as e:
                logger.error(f"Failed to open browser: {e}")
                print(f"Report generated at: {abs_report_path}")
                print("Please open it manually in your browser.")

        return str(report_path)

    except Exception as e:
        logger.error(f"Error generating backtest report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to generate backtest report: {str(e)}")