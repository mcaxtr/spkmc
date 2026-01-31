"""
Result export for the SPKMC algorithm.

This module contains functions to export simulation results in different formats,
such as CSV, Excel, JSON, Markdown, and HTML.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np

from spkmc.io.results import NumpyJSONEncoder

# Conditional imports
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not found. Excel/CSV export will be unavailable.")

from spkmc.visualization.plots import Visualizer


class ExportManager:
    """Manage result exports in different formats."""

    @staticmethod
    def export_to_csv(result: Dict[str, Any], output_path: str) -> str:
        """
        Export results to a CSV file.

        Args:
            result: Dictionary with results
            output_path: Output file path

        Returns:
            Path to the exported file

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not installed. Use 'pip install pandas' to export CSV.")

        # Create the directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )

        # Extract data
        time_steps = np.array(result.get("time", []))
        s_vals = np.array(result.get("S_val", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))

        # Create DataFrame
        data = {"Time": time_steps, "Susceptible": s_vals, "Infected": i_vals, "Recovered": r_vals}

        # Add error data if available
        if "S_err" in result and "I_err" in result and "R_err" in result:
            data["Susceptible_Error"] = np.array(result.get("S_err", []))
            data["Infected_Error"] = np.array(result.get("I_err", []))
            data["Recovered_Error"] = np.array(result.get("R_err", []))

        df = pd.DataFrame(data)

        # Save as CSV
        df.to_csv(output_path, index=False)

        return output_path

    @staticmethod
    def export_to_excel(result: Dict[str, Any], output_path: str) -> str:
        """
        Export results to an Excel file.

        Args:
            result: Dictionary with results
            output_path: Output file path

        Returns:
            Path to the exported file

        Raises:
            ImportError: If pandas or openpyxl are not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not installed. Use 'pip install pandas openpyxl' for Excel.")

        # Check whether openpyxl is available
        try:
            import openpyxl

            _ = openpyxl  # Verify import succeeded
        except ImportError:
            raise ImportError("Openpyxl not installed. Use 'pip install openpyxl' for Excel.")

        # Create the directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )

        # Extract data
        time_steps = np.array(result.get("time", []))
        s_vals = np.array(result.get("S_val", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))

        # Create the DataFrame for data
        data = {"Time": time_steps, "Susceptible": s_vals, "Infected": i_vals, "Recovered": r_vals}

        # Add error data if available
        if "S_err" in result and "I_err" in result and "R_err" in result:
            data["Susceptible_Error"] = np.array(result.get("S_err", []))
            data["Infected_Error"] = np.array(result.get("I_err", []))
            data["Recovered_Error"] = np.array(result.get("R_err", []))

        df_data = pd.DataFrame(data)

        # Create the DataFrame for metadata
        metadata = result.get("metadata", {})
        df_metadata = pd.DataFrame(list(metadata.items()), columns=["Parameter", "Value"])

        # Create the Excel file with multiple sheets
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_data.to_excel(writer, sheet_name="Data", index=False)
            df_metadata.to_excel(writer, sheet_name="Metadata", index=False)

            # Add a statistics sheet
            stats = {
                "Statistic": ["Max Infected", "Final Recovered", "Time to Peak"],
                "Value": [
                    np.max(i_vals),
                    r_vals[-1] if len(r_vals) > 0 else 0,
                    time_steps[np.argmax(i_vals)] if len(i_vals) > 0 else 0,
                ],
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name="Statistics", index=False)

        return output_path

    @staticmethod
    def export_to_json(result: Dict[str, Any], output_path: str) -> str:
        """
        Export results to a JSON file.

        Args:
            result: Dictionary with results
            output_path: Output file path

        Returns:
            Path to the exported file
        """
        # Create the directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

        return output_path

    @staticmethod
    def export_to_markdown(
        result: Dict[str, Any], output_path: str, include_plot: bool = True
    ) -> str:
        """
        Export results to a Markdown file.

        Args:
            result: Dictionary with results
            output_path: Output file path
            include_plot: If True, include a plot in the report

        Returns:
            Path to the exported file
        """
        # Create the directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )

        # Extract metadata
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")

        # Extract data
        time_steps = np.array(result.get("time", []))
        s_vals = np.array(result.get("S_val", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))

        # Calculate statistics
        max_infected = np.max(i_vals) if len(i_vals) > 0 else 0
        max_infected_time = (
            time_steps[np.argmax(i_vals)] if len(i_vals) > 0 and len(time_steps) > 0 else 0
        )
        final_recovered = r_vals[-1] if len(r_vals) > 0 else 0

        # Build Markdown content
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md_content = f"""# SPKMC Simulation Report

Generated at: {timestamp}

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Network Type | {network_type} |
| Distribution | {dist_type} |
| Number of Nodes (N) | {N} |
"""

        # Add specific parameters
        for key, value in metadata.items():
            if key not in ["network_type", "distribution", "N"]:
                md_content += f"| {key} | {value} |\n"

        # Add statistics
        md_content += f"""
## Statistics

| Statistic | Value |
|-------------|-------|
| Peak Infected | {max_infected:.4f} |
| Time to Infection Peak | {max_infected_time:.4f} |
| Final Recovered | {final_recovered:.4f} |

"""

        # Add plot if requested
        if include_plot:
            plot_path = output_path.replace(".md", ".png")

            # Generate plot
            title = f"SPKMC Simulation - Network {network_type}, Distribution {dist_type}, N={N}"

            has_error = "S_err" in result and "I_err" in result and "R_err" in result
            if has_error:
                s_err = np.array(result.get("S_err", []))
                i_err = np.array(result.get("I_err", []))
                r_err = np.array(result.get("R_err", []))
                Visualizer.plot_result_with_error(
                    s_vals, i_vals, r_vals, s_err, i_err, r_err, time_steps, title, plot_path
                )
            else:
                Visualizer.plot_result(s_vals, i_vals, r_vals, time_steps, title, plot_path)

            # Add plot reference to Markdown
            md_content += f"""
## Visualization

![Simulation Plot]({os.path.basename(plot_path)})

"""

        # Add data tables (first and last 5 points)
        md_content += """
## Simulation Data

### First 5 points

| Time | Susceptible | Infected | Recovered |
|-------|-------------|------------|-------------|
"""

        for idx in range(min(5, len(time_steps))):
            md_content += (
                f"| {time_steps[idx]:.4f} | {s_vals[idx]:.4f} "
                f"| {i_vals[idx]:.4f} | {r_vals[idx]:.4f} |\n"
            )

        md_content += """
### Last 5 points

| Time | Susceptible | Infected | Recovered |
|-------|-------------|------------|-------------|
"""

        for idx in range(max(0, len(time_steps) - 5), len(time_steps)):
            md_content += (
                f"| {time_steps[idx]:.4f} | {s_vals[idx]:.4f} "
                f"| {i_vals[idx]:.4f} | {r_vals[idx]:.4f} |\n"
            )

        # Save Markdown file
        with open(output_path, "w") as f:
            f.write(md_content)

        return output_path

    @staticmethod
    def export_to_html(result: Dict[str, Any], output_path: str, include_plot: bool = True) -> str:
        """
        Export results to an HTML file.

        Args:
            result: Dictionary with results
            output_path: Output file path
            include_plot: If True, include a plot in the report

        Returns:
            Path to the exported file

        Raises:
            ImportError: If pandas is not available
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas not installed. Use 'pip install pandas' to export HTML.")

        # First export to Markdown
        md_path = output_path.replace(".html", ".md")
        ExportManager.export_to_markdown(result, md_path, include_plot)

        # Convert Markdown to HTML using pandas
        with open(md_path, "r") as f:
            md_content = f.read()

        # Create a DataFrame with the Markdown content
        df = pd.DataFrame({"markdown": [md_content]})

        # Convert to HTML
        html = df.to_html(escape=False, index=False, header=False)

        # Add CSS styles
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPKMC Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""

        # Save HTML file
        with open(output_path, "w") as f:
            f.write(html_content)

        # Remove temporary Markdown file
        os.remove(md_path)

        return output_path

    @staticmethod
    def export_plot(
        result: Dict[str, Any], output_path: str, format: str = "png", dpi: int = 300
    ) -> str:
        """
        Export the simulation plot in different formats.

        Args:
            result: Dictionary with results
            output_path: Output file path
            format: Plot format (png, pdf, svg, jpg)
            dpi: Plot resolution in DPI

        Returns:
            Path to the exported file
        """
        # Create the directory if it doesn't exist
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True
        )

        # Extract metadata
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")

        # Extract data
        time_steps = np.array(result.get("time", []))
        s_vals = np.array(result.get("S_val", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))

        # Generate plot
        title = f"SPKMC Simulation - Network {network_type}, Distribution {dist_type}, N={N}"

        has_error = "S_err" in result and "I_err" in result and "R_err" in result
        if has_error:
            s_err = np.array(result.get("S_err", []))
            i_err = np.array(result.get("I_err", []))
            r_err = np.array(result.get("R_err", []))
            Visualizer.plot_result_with_error(
                s_vals, i_vals, r_vals, s_err, i_err, r_err, time_steps, title, output_path
            )
        else:
            Visualizer.plot_result(s_vals, i_vals, r_vals, time_steps, title, output_path)

        return output_path

    @staticmethod
    def export_results(result: Dict[str, Any], output_path: str, format: str = "json") -> str:
        """
        Export results in the specified format.

        Args:
            result: Dictionary with results
            output_path: Output file path
            format: Export format (json, csv, excel, md, html)

        Returns:
            Path to the exported file

        Raises:
            ValueError: If the format is invalid
            ImportError: If required dependencies are unavailable
        """
        format = format.lower()

        try:
            if format == "json":
                return ExportManager.export_to_json(result, output_path)
            elif format == "csv":
                if not PANDAS_AVAILABLE:
                    raise ImportError("Pandas not installed. Use 'pip install pandas' for CSV.")
                return ExportManager.export_to_csv(result, output_path)
            elif format == "excel":
                if not PANDAS_AVAILABLE:
                    raise ImportError(
                        "Pandas not installed. Use 'pip install pandas openpyxl' for Excel."
                    )
                return ExportManager.export_to_excel(result, output_path)
            elif format == "md" or format == "markdown":
                return ExportManager.export_to_markdown(result, output_path)
            elif format == "html":
                if not PANDAS_AVAILABLE:
                    raise ImportError("Pandas not installed. Use 'pip install pandas' for HTML.")
                return ExportManager.export_to_html(result, output_path)
            else:
                raise ValueError(f"Invalid export format: {format}")
        except ImportError as e:
            print(f"Import error: {e}")
            print("Trying to export to JSON as a fallback...")
            json_path = output_path.replace(f".{format}", ".json")
            return ExportManager.export_to_json(result, json_path)
