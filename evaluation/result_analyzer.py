import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyze and visualize site selection evaluation results."""

    def __init__(self, results_dir: str):
        """
        Initialize with the directory containing evaluation results.

        Args:
            results_dir: Directory containing result files
        """
        self.results_dir = results_dir
        self.summary = None
        self.detailed_results = {}

        # Load the results
        self.load_results()

    def load_results(self):
        """Load all result files from the results directory."""
        try:
            # Load summary if it exists
            summary_path = os.path.join(self.results_dir, "summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    self.summary = json.load(f)
                logger.info(f"Loaded summary from {summary_path}")

            # Load detailed results for each method
            method_files = {
                "zero_shot": "zero_shot_results.json",
                "few_shot": "few_shot_results.json",
                "fine_tuned": "fine_tuned_results.json",
                "fine_tuned_rag": "fine_tuned_rag_results.json"  # Add this line
            }

            for method, filename in method_files.items():
                file_path = os.path.join(self.results_dir, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        self.detailed_results[method] = json.load(f)
                    logger.info(f"Loaded {method} results from {file_path}")

            # Create summary if it doesn't exist yet
            if not self.summary and self.detailed_results:
                self.create_summary()

        except Exception as e:
            logger.error(f"Error loading results: {e}")

    def create_summary(self):
        """Create a summary of evaluation results from detailed results."""
        if not self.detailed_results:
            logger.warning("No detailed results available to create summary")
            return

        summary = {}

        for method, results in self.detailed_results.items():
            # Filter successful results
            successful_results = [
                r for r in results if r.get('success', False)]

            if not successful_results:
                logger.warning(f"No successful results for method: {method}")
                continue

            # Calculate metrics
            total_count = len(results)
            success_count = len(successful_results)
            success_rate = success_count / total_count if total_count > 0 else 0.0

            # Calculate averages for successful results
            avg_metrics = {
                'avg_precision': sum(r.get('precision', 0.0) for r in successful_results) / max(success_count, 1),
                'avg_recall': sum(r.get('recall', 0.0) for r in successful_results) / max(success_count, 1),
                'avg_f1_score': sum(r.get('f1_score', 0.0) for r in successful_results) / max(success_count, 1),
                'avg_exact_match': sum(r.get('exact_match', 0.0) for r in successful_results) / max(success_count, 1),
                'avg_result_match_rate': sum(r.get('result_match_rate', 0.0) for r in successful_results) / max(success_count, 1)
            }

            # Create summary entry
            summary[method] = {
                'total_queries': total_count,
                'successful_queries': success_count,
                'success_rate': success_rate,
                **avg_metrics
            }

        # Store the summary
        self.summary = summary

        # Save summary to file
        summary_path = os.path.join(self.results_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Created and saved summary to {summary_path}")

        return summary

    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of evaluation metrics.

        Returns:
            DataFrame with summary metrics
        """
        if not self.summary:
            logger.warning("No summary data available")
            return pd.DataFrame()

        # Convert summary to DataFrame
        summary_df = pd.DataFrame.from_dict(self.summary, orient='index')

        # Format percentages
        for col in summary_df.columns:
            if col.startswith('avg_') or col == 'success_rate':
                summary_df[col] = summary_df[col].map(lambda x: f"{x:.2%}")

        return summary_df

    def plot_metric_comparison(self, output_file: Optional[str] = None):
        """
        Create a bar chart comparing metrics across methods.

        Args:
            output_file: Path to save the plot (None to display)
        """
        if not self.summary:
            logger.warning("No summary data available")
            return

        # Extract metrics of interest
        metrics = ['avg_precision', 'avg_recall',
                   'avg_f1_score', 'avg_exact_match', 'success_rate']
        method_names = list(self.summary.keys())

        # Create a DataFrame for plotting
        plot_data = {}
        for method in method_names:
            plot_data[method] = {
                'Precision': self.summary[method]['avg_precision'],
                'Recall': self.summary[method]['avg_recall'],
                'F1 Score': self.summary[method]['avg_f1_score'],
                'Exact Match': self.summary[method]['avg_exact_match'],
                'Success Rate': self.summary[method]['success_rate']
            }
        plot_df = pd.DataFrame(plot_data).T

        # Set up the plot
        plt.figure(figsize=(12, 8))
        ax = plot_df.plot(kind='bar', width=0.8, figsize=(12, 8))

        # Add labels and title
        plt.title('Performance Comparison of Different Methods', fontsize=16)
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10)

        # Add legend
        plt.legend(title='Metrics', title_fontsize=12,
                   fontsize=10, loc='upper right')

        plt.tight_layout()

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

    def analyze_query_performance(self) -> pd.DataFrame:
        """
        Analyze performance by query across methods.

        Returns:
            DataFrame with query-level metrics
        """
        if not self.detailed_results:
            logger.warning("No detailed results available")
            return pd.DataFrame()

        # Create a DataFrame to hold query-level metrics
        query_metrics = []

        for method, results in self.detailed_results.items():
            for result in results:
                if result.get('success', False):
                    query_metrics.append({
                        'method': method,
                        'query': result['query'],
                        'precision': result.get('precision', 0),
                        'recall': result.get('recall', 0),
                        'f1_score': result.get('f1_score', 0),
                        'exact_match': result.get('exact_match', 0),
                        'result_match_rate': result.get('result_match_rate', 0),
                        'generated_count': result.get('generated_count', 0),
                        'ground_truth_count': result.get('ground_truth_count', 0),
                        'correct_count': result.get('correct_count', 0)
                    })

        # Convert to DataFrame
        query_df = pd.DataFrame(query_metrics)

        # If empty, return early
        if query_df.empty:
            return query_df

        # Categorize queries by type
        query_df['query_type'] = query_df['query'].apply(self.categorize_query)

        return query_df

    def categorize_query(self, query: str) -> str:
        """
        Categorize a query based on its content.

        Args:
            query: Query text

        Returns:
            Category string
        """
        query_lower = query.lower()

        if 'commercial' in query_lower and not any(term in query_lower for term in ['retail', 'office', 'mixed-use', 'vacant']):
            return 'Commercial'
        elif 'retail' in query_lower:
            return 'Retail'
        elif 'office' in query_lower:
            return 'Office'
        elif 'mixed-use' in query_lower or 'mixed use' in query_lower:
            return 'Mixed-Use'
        elif 'vacant' in query_lower:
            return 'Vacant'
        elif 'residential' in query_lower:
            return 'Residential'
        elif 'top' in query_lower:
            return 'Top N'
        elif any(constraint in query_lower for constraint in ['larger than', 'smaller than', 'between', 'at least', 'sq ft', 'square feet']):
            return 'Size Constraint'
        elif any(constraint in query_lower for constraint in ['within', 'near', 'distance', 'meters of']):
            return 'Proximity Constraint'
        else:
            return 'Other'

    def plot_query_type_performance(self, output_file: Optional[str] = None):
        """
        Create a bar chart showing performance by query type across methods.

        Args:
            output_file: Path to save the plot (None to display)
        """
        query_df = self.analyze_query_performance()

        if query_df.empty:
            logger.warning("No query performance data available")
            return

        # Calculate mean F1 score by method and query type
        performance_by_type = query_df.groupby(['method', 'query_type'])[
            'f1_score'].mean().unstack()

        # Plot
        plt.figure(figsize=(14, 8))
        ax = performance_by_type.plot(kind='bar', width=0.8, figsize=(14, 8))

        # Add labels and title
        plt.title('F1 Score by Query Type and Method', fontsize=16)
        plt.xlabel('Method', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add legend
        plt.legend(title='Query Type', title_fontsize=12,
                   fontsize=10, loc='upper right')

        plt.tight_layout()

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

    def analyze_code_complexity(self) -> pd.DataFrame:
        """
        Analyze code complexity across methods.

        Returns:
            DataFrame with code complexity metrics
        """
        if not self.detailed_results:
            logger.warning("No detailed results available")
            return pd.DataFrame()

        # Create a DataFrame to hold complexity metrics
        complexity_metrics = []

        for method, results in self.detailed_results.items():
            for result in results:
                if result.get('success', False) and 'code' in result:
                    code = result['code']

                    # Calculate complexity metrics
                    lines = len(code.split('\n'))
                    chars = len(code)
                    functions = code.count('def ')

                    complexity_metrics.append({
                        'method': method,
                        'query': result['query'],
                        'lines_of_code': lines,
                        'characters': chars,
                        'functions': functions,
                        'f1_score': result.get('f1_score', 0)
                    })

        # Convert to DataFrame
        complexity_df = pd.DataFrame(complexity_metrics)

        return complexity_df

    def plot_complexity_vs_performance(self, output_file: Optional[str] = None):
        """
        Create a scatter plot showing the relationship between code complexity and performance.

        Args:
            output_file: Path to save the plot (None to display)
        """
        complexity_df = self.analyze_code_complexity()

        if complexity_df.empty:
            logger.warning("No code complexity data available")
            return

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Create scatter plot for each method
        methods = complexity_df['method'].unique()
        markers = ['o', 's', '^', 'D']
        colors = ['blue', 'green', 'red', 'purple']

        for i, method in enumerate(methods):
            method_data = complexity_df[complexity_df['method'] == method]
            plt.scatter(
                method_data['lines_of_code'],
                method_data['f1_score'],
                label=method,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                alpha=0.7,
                s=80
            )

        # Add trend line for all data
        x = complexity_df['lines_of_code']
        y = complexity_df['f1_score']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "k--", alpha=0.8, label="Trend")

        # Add correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]
        plt.text(
            0.05, 0.05,
            f"Correlation: {correlation:.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        # Add labels and title
        plt.title(
            'Relationship Between Code Complexity and Performance', fontsize=16)
        plt.xlabel('Lines of Code', fontsize=14)
        plt.ylabel('F1 Score', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 1.1)
        plt.xlim(0, None)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        plt.legend(title='Method', title_fontsize=12,
                   fontsize=10, loc='upper right')

        plt.tight_layout()

        # Save or show the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()

    def analyze_error_patterns(self) -> pd.DataFrame:
        """
        Analyze patterns in code generation and execution errors.

        Returns:
            DataFrame with error analysis
        """
        if not self.detailed_results:
            logger.warning("No detailed results available")
            return pd.DataFrame()

        # Create a DataFrame to hold error metrics
        error_metrics = []

        for method, results in self.detailed_results.items():
            # Count total queries
            total_queries = len(results)

            # Count successful executions
            successful = sum(1 for r in results if r.get('success', False))

            # Count failed executions
            failed = total_queries - successful

            # Analyze specific error patterns in code
            code_errors = []
            for result in results:
                if not result.get('success', False) and 'code' in result:
                    code = result['code']

                    # Look for common error patterns
                    if "ImportError" in result.get('error', ''):
                        code_errors.append("Missing import")
                    elif "SyntaxError" in result.get('error', ''):
                        code_errors.append("Syntax error")
                    elif "FileNotFoundError" in result.get('error', ''):
                        code_errors.append("File not found")
                    elif "KeyError" in result.get('error', ''):
                        code_errors.append("Key error")
                    elif "AttributeError" in result.get('error', ''):
                        code_errors.append("Attribute error")
                    else:
                        code_errors.append("Other error")

            # Count error types
            error_counts = {}
            for error in code_errors:
                if error in error_counts:
                    error_counts[error] += 1
                else:
                    error_counts[error] = 1

            # Add to metrics
            error_metrics.append({
                'method': method,
                'total_queries': total_queries,
                'successful': successful,
                'failed': failed,
                'success_rate': successful / total_queries if total_queries > 0 else 0,
                'error_types': error_counts
            })

        # Convert to DataFrame
        error_df = pd.DataFrame(error_metrics)

        return error_df

    def generate_report(self, output_dir: Optional[str] = None):
        """
        Generate a comprehensive analysis report with visualizations.

        Args:
            output_dir: Directory to save the report files (None to use results_dir)
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, 'analysis')

        os.makedirs(output_dir, exist_ok=True)

        # Ensure we have a summary
        if not self.summary and self.detailed_results:
            self.create_summary()

        # Generate summary table
        summary_df = self.create_summary_table()
        summary_path = os.path.join(output_dir, 'summary_table.csv')
        summary_df.to_csv(summary_path)
        logger.info(f"Saved summary table to {summary_path}")

        # Generate plots
        metric_plot_path = os.path.join(output_dir, 'metric_comparison.png')
        self.plot_metric_comparison(metric_plot_path)

        query_plot_path = os.path.join(
            output_dir, 'query_type_performance.png')
        self.plot_query_type_performance(query_plot_path)

        complexity_plot_path = os.path.join(
            output_dir, 'complexity_vs_performance.png')
        self.plot_complexity_vs_performance(complexity_plot_path)

        # Generate query performance analysis
        query_df = self.analyze_query_performance()
        query_path = os.path.join(output_dir, 'query_performance.csv')
        query_df.to_csv(query_path, index=False)
        logger.info(f"Saved query performance analysis to {query_path}")

        # Generate code complexity analysis
        complexity_df = self.analyze_code_complexity()
        complexity_path = os.path.join(output_dir, 'code_complexity.csv')
        complexity_df.to_csv(complexity_path, index=False)
        logger.info(f"Saved code complexity analysis to {complexity_path}")

        # Generate error analysis
        error_df = self.analyze_error_patterns()
        error_path = os.path.join(output_dir, 'error_analysis.csv')
        error_df.to_csv(error_path, index=False)
        logger.info(f"Saved error analysis to {error_path}")

        # Generate HTML report
        self.generate_html_report(output_dir)

        logger.info(f"Analysis report generated in {output_dir}")

        # Return summary for quick reference
        return summary_df

    def generate_html_report(self, output_dir: str):
        """
        Generate an HTML report of evaluation results.

        Args:
            output_dir: Directory to save the HTML report
        """
        try:
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Site Selection Method Evaluation Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metric-image {{ max-width: 100%; height: auto; margin: 20px 0; }}
                    .summary {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Site Selection Method Evaluation Report</h1>
                    
                    <div class="summary">
                        <h2>Performance Summary</h2>
                        <table>
                            <tr>
                                <th>Method</th>
                                <th>Success Rate</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>Exact Match</th>
                            </tr>
            """

            # Add summary rows
            if self.summary:
                for method, metrics in self.summary.items():
                    html_content += f"""
                    <tr>
                        <td>{method}</td>
                        <td>{metrics['success_rate']:.2%}</td>
                        <td>{metrics['avg_precision']:.2%}</td>
                        <td>{metrics['avg_recall']:.2%}</td>
                        <td>{metrics['avg_f1_score']:.2%}</td>
                        <td>{metrics['avg_exact_match']:.2%}</td>
                    </tr>
                    """

            html_content += """
                        </table>
                    </div>
                    
                    <div class="visualizations">
                        <h2>Performance Visualizations</h2>
                        
                        <h3>Method Comparison</h3>
                        <img src="metric_comparison.png" alt="Method Comparison" class="metric-image">
                        
                        <h3>Performance by Query Type</h3>
                        <img src="query_type_performance.png" alt="Query Type Performance" class="metric-image">
                        
                        <h3>Code Complexity vs. Performance</h3>
                        <img src="complexity_vs_performance.png" alt="Complexity vs Performance" class="metric-image">
                    </div>
                    
                    <div class="conclusions">
                        <h2>Conclusions</h2>
                        <p>Based on the evaluation results, the following conclusions can be drawn:</p>
                        <ul>
            """

            # Add conclusions based on results
            best_method = None
            best_f1 = -1

            if self.summary:
                for method, metrics in self.summary.items():
                    if metrics['avg_f1_score'] > best_f1:
                        best_f1 = metrics['avg_f1_score']
                        best_method = method

                html_content += f"<li>The best-performing method is <strong>{best_method}</strong> with an F1 score of {best_f1:.2%}.</li>"

                # Add more specific insights
                query_df = self.analyze_query_performance()
                if not query_df.empty:
                    # Find the query type with highest F1 score
                    best_query_type = query_df.groupby(
                        'query_type')['f1_score'].mean().idxmax()
                    best_query_f1 = query_df.groupby(
                        'query_type')['f1_score'].mean().max()

                    html_content += f"<li>The methods perform best on <strong>{best_query_type}</strong> queries with an average F1 score of {best_query_f1:.2%}.</li>"

                    # Find the query type with most variance
                    query_variance = query_df.groupby(
                        'query_type')['f1_score'].std()
                    if not query_variance.empty:
                        most_variable = query_variance.idxmax()
                        variance_value = query_variance.max()

                        html_content += f"<li>The highest variability in performance is seen in <strong>{most_variable}</strong> queries with a standard deviation of {variance_value:.2%}.</li>"

            html_content += """
                        </ul>
                    </div>
                </div>
            </body>
            </html>
            """

            # Save HTML file
            html_path = os.path.join(output_dir, "evaluation_report.html")
            with open(html_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Generated HTML report at {html_path}")

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
