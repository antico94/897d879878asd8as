import numpy as np
from ReportGeneration.ReportUtils.html_utils import get_text_class, embed_image


def generate_accuracy_section(backtest_results):
    """Generate the prediction accuracy analysis section."""
    # Extract prediction data if available
    predictions = backtest_results.get('predictions', {})

    if not predictions:
        return '<div class="card"><div class="card-header">Prediction Accuracy Analysis</div><div class="card-body"><p>No prediction data available for analysis.</p></div></div>'

    # Extract accuracy metrics
    overall_accuracy = predictions.get('overall_accuracy', 0)
    class_accuracy = predictions.get('class_accuracy', {})
    confusion_matrix = predictions.get('confusion_matrix', [])
    confusion_matrix_plot = predictions.get('confusion_matrix_plot', '')

    # Extract confidence data if available
    confidence_data = predictions.get('confidence', {})
    confidence_plot = predictions.get('confidence_plot', '')

    # Define thresholds for color coding
    accuracy_thresholds = {'good': 0.60, 'average': 0.53}

    # Create overall accuracy metrics
    accuracy_html = f"""
    <div class="metric">
        <div class="metric-label">Overall Prediction Accuracy:</div>
        <div class="metric-value {get_text_class(overall_accuracy, accuracy_thresholds)}">{overall_accuracy:.2%}</div>
    </div>
    """

    # Add class-specific metrics
    for class_id, class_data in class_accuracy.items():
        if isinstance(class_data, dict):
            acc = class_data.get('accuracy', 0)
            samples = class_data.get('samples', 0)

            # Determine label based on class_id (assuming 1 = UP, 0 = DOWN)
            label = "UP" if str(class_id) == "1" or class_id == 1 else "DOWN"

            # Add to metrics
            accuracy_html += f"""
            <div class="metric">
                <div class="metric-label">Accuracy for {label}:</div>
                <div class="metric-value {get_text_class(acc, accuracy_thresholds)}">{acc:.2%} ({samples} samples)</div>
            </div>
            """

    # Embed confusion matrix plot if available
    confusion_matrix_html = ""
    if confusion_matrix_plot:
        embedded_img = embed_image(confusion_matrix_plot)
        confusion_matrix_html = f"""
        <div class="chart-container">
            <h3>Confusion Matrix</h3>
            <img src="{embedded_img}" alt="Confusion Matrix">
            <div class="card-explanation">
                <p><strong>How to interpret:</strong> The confusion matrix shows the counts of correct and incorrect predictions. The vertical axis shows the actual values, while the horizontal axis shows the predicted values.</p>
                <ul>
                    <li>Top-left: True Negatives (correctly predicted DOWN)</li>
                    <li>Top-right: False Positives (incorrectly predicted UP)</li>
                    <li>Bottom-left: False Negatives (incorrectly predicted DOWN)</li>
                    <li>Bottom-right: True Positives (correctly predicted UP)</li>
                </ul>
            </div>
        </div>
        """
    # Display raw confusion matrix if no plot but matrix data available
    elif confusion_matrix:
        try:
            # Create simple textual representation
            if len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2:
                tn, fp = confusion_matrix[0]
                fn, tp = confusion_matrix[1]

                confusion_matrix_html = f"""
                <div class="chart-container">
                    <h3>Confusion Matrix</h3>
                    <table style="width: 300px; margin: 0 auto;">
                        <tr>
                            <td></td>
                            <td style="text-align: center;"><strong>Predicted DOWN</strong></td>
                            <td style="text-align: center;"><strong>Predicted UP</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Actual DOWN</strong></td>
                            <td style="text-align: center; background-color: rgba(46, 204, 113, 0.3);">{tn}</td>
                            <td style="text-align: center; background-color: rgba(231, 76, 60, 0.3);">{fp}</td>
                        </tr>
                        <tr>
                            <td><strong>Actual UP</strong></td>
                            <td style="text-align: center; background-color: rgba(231, 76, 60, 0.3);">{fn}</td>
                            <td style="text-align: center; background-color: rgba(46, 204, 113, 0.3);">{tp}</td>
                        </tr>
                    </table>
                    <div class="card-explanation">
                        <p><strong>How to interpret:</strong> The confusion matrix shows the counts of correct and incorrect predictions.</p>
                        <ul>
                            <li>Correctly predicted DOWN (True Negative): {tn}</li>
                            <li>Incorrectly predicted UP when actually DOWN (False Positive): {fp}</li>
                            <li>Incorrectly predicted DOWN when actually UP (False Negative): {fn}</li>
                            <li>Correctly predicted UP (True Positive): {tp}</li>
                        </ul>
                    </div>
                </div>
                """
        except Exception:
            # Fallback if matrix format is not as expected
            confusion_matrix_html = ""

    # Create confidence analysis if available
    confidence_html = ""
    if confidence_data:
        # Extract confidence stats
        confidence_stats = confidence_data.get('stats', {})
        confidence_by_level = confidence_data.get('by_level', {})

        # Create confidence stats HTML
        if confidence_stats:
            confidence_html = f"""
            <h3>Prediction Confidence Analysis</h3>
            <div class="metric">
                <div class="metric-label">Average Confidence:</div>
                <div class="metric-value">{confidence_stats.get('mean', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Minimum Confidence:</div>
                <div class="metric-value">{confidence_stats.get('min', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Maximum Confidence:</div>
                <div class="metric-value">{confidence_stats.get('max', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Standard Deviation:</div>
                <div class="metric-value">{confidence_stats.get('std', 0):.4f}</div>
            </div>
            """

            # Create confidence by level table if available
            if confidence_by_level:
                confidence_html += """
                <h3>Accuracy by Confidence Level</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Confidence Range</th>
                            <th>Accuracy</th>
                            <th>Samples</th>
                            <th>% of Total</th>
                        </tr>
                    </thead>
                    <tbody>
                """

                # Sort confidence levels
                sorted_levels = sorted(confidence_by_level.keys(), key=lambda x: float(x.split(',')[0].strip('()')))

                for level in sorted_levels:
                    level_data = confidence_by_level[level]
                    accuracy = level_data.get('accuracy', 0)
                    samples = level_data.get('samples', 0)
                    percentage = level_data.get('percentage', 0)

                    # Get appropriate color class
                    accuracy_class = get_text_class(accuracy, accuracy_thresholds)

                    confidence_html += f"""
                    <tr>
                        <td>{level}</td>
                        <td class="{accuracy_class}">{accuracy:.2%}</td>
                        <td>{samples}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
                    """

                confidence_html += """
                    </tbody>
                </table>
                """

        # Embed confidence plot if available
        if confidence_plot:
            embedded_img = embed_image(confidence_plot)
            confidence_html += f"""
            <div class="chart-container">
                <h3>Accuracy by Confidence Level</h3>
                <img src="{embedded_img}" alt="Accuracy by Confidence Level">
                <div class="card-explanation">
                    <p><strong>How to interpret:</strong> This chart shows how prediction accuracy relates to confidence levels. Ideally, higher confidence predictions should have higher accuracy.</p>
                </div>
            </div>
            """

    # Create recommendations based on accuracy analysis
    accuracy_recommendations = []

    # Check overall accuracy
    if overall_accuracy < 0.53:
        accuracy_recommendations.append(
            "The overall accuracy is below 53%, which is only slightly better than random guessing. Consider retraining the model or adjusting the trading strategy.")
    elif overall_accuracy >= 0.6:
        accuracy_recommendations.append(
            f"The strong overall accuracy of {overall_accuracy:.2%} indicates good predictive power.")

    # Check for bias in prediction accuracy
    if class_accuracy:
        up_accuracy = class_accuracy.get("1", class_accuracy.get(1, {})).get('accuracy', 0)
        down_accuracy = class_accuracy.get("0", class_accuracy.get(0, {})).get('accuracy', 0)

        if abs(up_accuracy - down_accuracy) > 0.15:
            if up_accuracy > down_accuracy:
                accuracy_recommendations.append(
                    f"The model is much better at predicting UP movements ({up_accuracy:.2%}) than DOWN movements ({down_accuracy:.2%}). Consider focusing on long trades.")
            else:
                accuracy_recommendations.append(
                    f"The model is much better at predicting DOWN movements ({down_accuracy:.2%}) than UP movements ({up_accuracy:.2%}). Consider focusing on short trades.")

    # Check confidence-accuracy relationship
    if confidence_by_level:
        highest_conf_level = sorted(confidence_by_level.keys(), key=lambda x: float(x.split(',')[0].strip('()')))[-1]
        highest_conf_acc = confidence_by_level[highest_conf_level].get('accuracy', 0)
        lowest_conf_level = sorted(confidence_by_level.keys(), key=lambda x: float(x.split(',')[0].strip('()')))[0]
        lowest_conf_acc = confidence_by_level[lowest_conf_level].get('accuracy', 0)

        if highest_conf_acc - lowest_conf_acc < 0.1:
            accuracy_recommendations.append(
                "Confidence levels don't strongly correlate with accuracy. The model may be miscalibrated.")
        elif highest_conf_acc > 0.65:
            accuracy_recommendations.append(
                f"High-confidence predictions ({highest_conf_level}) show strong accuracy ({highest_conf_acc:.2%}). Consider filtering trades to only use signals with higher confidence.")

    # Create recommendations HTML
    recommendations_html = ""
    if accuracy_recommendations:
        recommendations_html = """
        <div class="recommendations">
            <h3>Prediction Accuracy Insights</h3>
            <ul>
        """
        for rec in accuracy_recommendations:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += """
            </ul>
        </div>
        """

    # Create explanation of accuracy
    accuracy_explanation = """
    <div class="card-explanation">
        <p><strong>What is Prediction Accuracy?</strong> In the context of trading, prediction accuracy measures how often the model correctly predicts price movement direction (up or down).</p>
        <p><strong>Why does it matter?</strong> While accuracy isn't the only factor for trading success (profit size matters too), it's a fundamental measure of the model's predictive power.</p>
        <p><strong>How to interpret confidence?</strong> Confidence represents how certain the model is about each prediction. Ideally, higher confidence should correlate with higher accuracy.</p>
    </div>
    """

    # Build the complete section
    html = f"""
    <div class="card">
        <div class="card-header">Prediction Accuracy Analysis</div>
        <div class="card-body">
            <h3>Prediction Accuracy</h3>
            {accuracy_html}

            {accuracy_explanation}

            {confusion_matrix_html}

            {confidence_html}

            {recommendations_html}
        </div>
    </div>
    """

    return html