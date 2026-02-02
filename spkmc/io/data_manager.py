"""
Unified data management for the SPKMC algorithm.

This module provides a single DataManager class for all data I/O operations,
supporting multiple formats (JSON, CSV, Excel, Markdown, HTML).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, cast

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: object) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class DataManager:
    """
    Simple data I/O for SPKMC simulations.

    This class provides unified save/load functionality for simulation results,
    auto-detecting format from file extension.

    Supported formats:
    - JSON (.json): Full result with metadata
    - CSV (.csv): Data table format
    - Excel (.xlsx): Multi-sheet with data, metadata, and statistics
    - Markdown (.md): Human-readable report
    - HTML (.html): Web-viewable report

    Usage:
        # Save result
        DataManager.save(result, "data/runs/gamma/ER/results.json")

        # Load result
        result = DataManager.load("data/runs/gamma/ER/results.json")

        # Check existence
        if DataManager.exists("data/runs/gamma/ER/results.json"):
            ...

        # List files
        files = DataManager.list_files("data/runs/gamma/ER", "*.json")
    """

    @classmethod
    def save(cls, result: Dict[str, Any], path: str) -> str:
        """
        Save result to file. Format auto-detected from extension.

        Args:
            result: Dict with SIR data (S_val, I_val, R_val, time, metadata)
            path: Full path where to save (e.g., "data/runs/gamma/ER/results.json")

        Returns:
            Path where file was saved

        Raises:
            ValueError: If format is not supported
        """
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        ext = Path(path).suffix.lower()

        if ext == ".json":
            cls._save_json(result, path)
        elif ext == ".csv":
            cls._save_csv(result, path)
        elif ext == ".xlsx":
            cls._save_excel(result, path)
        elif ext == ".md":
            cls._save_markdown(result, path)
        elif ext == ".html":
            cls._save_html(result, path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        return path

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        """
        Load result from file. Format auto-detected from extension.

        Args:
            path: Path to result file

        Returns:
            Result dict with SIR data and metadata

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If format is not supported
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = Path(path).suffix.lower()

        if ext == ".json":
            return cls._load_json(path)
        elif ext == ".csv":
            return cls._load_csv(path)
        elif ext in (".xlsx", ".xls"):
            return cls._load_excel(path)
        else:
            raise ValueError(f"Unsupported format for loading: {ext}")

    @classmethod
    def exists(cls, path: str) -> bool:
        """Check if file exists."""
        return Path(path).exists()

    @classmethod
    def list_files(cls, directory: str, pattern: str = "*.*") -> List[str]:
        """
        List files in directory matching pattern.

        Args:
            directory: Directory path to search
            pattern: Glob pattern (default: "*.*")

        Returns:
            Sorted list of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        return sorted(str(f) for f in dir_path.glob(pattern))

    @classmethod
    def list_all_results(cls, base_dir: str = "data/runs") -> List[str]:
        """
        List all result files under the base directory.

        Args:
            base_dir: Base directory for results (default: data/runs)

        Returns:
            List of all JSON result file paths
        """
        results: List[str] = []
        base_path = Path(base_dir)

        if not base_path.exists():
            return results

        # Supported result file extensions
        supported_extensions = ["*.json", "*.csv", "*.xlsx", "*.md", "*.html"]

        # Walk through distribution/network hierarchy
        for dist_dir in base_path.iterdir():
            if dist_dir.is_dir():
                for network_dir in dist_dir.iterdir():
                    if network_dir.is_dir():
                        for ext_pattern in supported_extensions:
                            for result_file in network_dir.glob(ext_pattern):
                                results.append(str(result_file))

        return sorted(results)

    # Private save methods

    @classmethod
    def _save_json(cls, result: Dict[str, Any], path: str) -> None:
        """Save result as JSON."""
        with open(path, "w") as f:
            json.dump(result, f, indent=2, cls=NumpyJSONEncoder)

    @classmethod
    def _save_csv(cls, result: Dict[str, Any], path: str) -> None:
        """Save result as CSV."""
        try:
            import pandas  # noqa: F401
        except ImportError:
            raise ImportError("Pandas is required for CSV export. Install with: pip install pandas")

        df = cls._result_to_dataframe(result)
        df.to_csv(path, index=False)

    @classmethod
    def _save_excel(cls, result: Dict[str, Any], path: str) -> None:
        """Save result as Excel with multiple sheets."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for Excel export. Install with: pip install pandas openpyxl"
            )

        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required for Excel export. Install with: pip install openpyxl"
            )

        df_data = cls._result_to_dataframe(result)

        # Create metadata DataFrame
        metadata = result.get("metadata", {})
        df_metadata = pd.DataFrame([{"Parameter": k, "Value": v} for k, v in metadata.items()])

        # Create statistics DataFrame
        time_steps = np.array(result.get("time", []))
        i_vals = np.array(result.get("I_val", []))
        r_vals = np.array(result.get("R_val", []))

        stats = {
            "Statistic": ["Max Infected", "Final Recovered", "Time to Peak"],
            "Value": [
                np.max(i_vals) if len(i_vals) > 0 else 0,
                r_vals[-1] if len(r_vals) > 0 else 0,
                time_steps[np.argmax(i_vals)] if len(i_vals) > 0 and len(time_steps) > 0 else 0,
            ],
        }
        df_stats = pd.DataFrame(stats)

        # Write to Excel
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_data.to_excel(writer, sheet_name="Data", index=False)
            df_metadata.to_excel(writer, sheet_name="Metadata", index=False)
            df_stats.to_excel(writer, sheet_name="Statistics", index=False)

    @classmethod
    def _save_markdown(cls, result: Dict[str, Any], path: str, include_plot: bool = True) -> None:
        """Save result as Markdown report."""
        # Extract metadata
        metadata = result.get("metadata", {})
        network_type = metadata.get("network", "").upper()
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
            if key not in ["network", "distribution", "N"]:
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

        # Add plot reference if requested
        if include_plot:
            plot_path = path.replace(".md", ".png")
            cls._generate_plot(result, plot_path)
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
        with open(path, "w") as f:
            f.write(md_content)

    @classmethod
    def _save_html(cls, result: Dict[str, Any], path: str, include_plot: bool = True) -> None:
        """Save result as HTML report."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for HTML export. Install with: pip install pandas"
            )

        # First export to Markdown
        md_path = path.replace(".html", "_temp.md")
        cls._save_markdown(result, md_path, include_plot)

        # Read markdown content
        with open(md_path, "r") as f:
            md_content = f.read()

        # Convert to simple HTML table
        df = pd.DataFrame({"markdown": [md_content]})
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
        with open(path, "w") as f:
            f.write(html_content)

        # Remove temporary Markdown file
        os.remove(md_path)

    @classmethod
    def _generate_plot(cls, result: Dict[str, Any], output_path: str) -> None:
        """Generate and save a plot for the result."""
        from spkmc.visualization.plots import Visualizer

        # Extract metadata
        metadata = result.get("metadata", {})
        network_type = metadata.get("network", "").upper()
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

    # Private load methods

    @classmethod
    def _load_json(cls, path: str) -> Dict[str, Any]:
        """Load result from JSON file."""
        with open(path, "r") as f:
            return cast(Dict[str, Any], json.load(f))

    @classmethod
    def _load_csv(cls, path: str) -> Dict[str, Any]:
        """Load result from CSV file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for CSV loading. Install with: pip install pandas"
            )

        df = pd.read_csv(path)
        return cls._dataframe_to_result(df, path)

    @classmethod
    def _load_excel(cls, path: str) -> Dict[str, Any]:
        """Load result from Excel file."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is required for Excel loading. Install with: pip install pandas openpyxl"
            )

        # Read the Data sheet
        try:
            df = pd.read_excel(path, sheet_name="Data")
        except ValueError:
            # Try reading the first sheet if "Data" doesn't exist
            df = pd.read_excel(path, sheet_name=0)

        result = cls._dataframe_to_result(df, path)

        # Try to load metadata from Metadata sheet
        try:
            df_meta = pd.read_excel(path, sheet_name="Metadata")
            if "Parameter" in df_meta.columns and "Value" in df_meta.columns:
                for _, row in df_meta.iterrows():
                    result["metadata"][row["Parameter"]] = row["Value"]
        except (ValueError, KeyError):
            pass  # Metadata sheet not found or invalid format

        return result

    # Helper methods

    @classmethod
    def _result_to_dataframe(cls, result: Dict[str, Any]) -> "pd.DataFrame":
        """Convert result dict to DataFrame."""
        import pandas as pd

        data = {
            "Time": result.get("time", []),
            "Susceptible": result.get("S_val", []),
            "Infected": result.get("I_val", []),
            "Recovered": result.get("R_val", []),
        }
        if "S_err" in result:
            data["Susceptible_Error"] = result["S_err"]
            data["Infected_Error"] = result["I_err"]
            data["Recovered_Error"] = result["R_err"]
        return pd.DataFrame(data)

    @classmethod
    def _dataframe_to_result(cls, df: "pd.DataFrame", source: str) -> Dict[str, Any]:
        """Convert DataFrame to result dict."""
        # Validate required columns
        required_columns = {"Time", "Susceptible", "Infected", "Recovered"}
        if not required_columns.issubset(df.columns):
            # Try alternative column names (lowercase after we lowercase the df columns)
            alt_columns = {"time", "s", "i", "r"}
            df_lower = df.copy()
            df_lower.columns = df_lower.columns.str.lower()
            if alt_columns.issubset(df_lower.columns):
                df = df_lower.rename(
                    columns={"s": "Susceptible", "i": "Infected", "r": "Recovered", "time": "Time"}
                )
            else:
                raise ValueError(
                    f"CSV must contain columns: {required_columns}. Found: {set(df.columns)}"
                )

        result: Dict[str, Any] = {
            "time": df["Time"].tolist(),
            "S_val": df["Susceptible"].tolist(),
            "I_val": df["Infected"].tolist(),
            "R_val": df["Recovered"].tolist(),
            "has_error": False,
            "metadata": {"source": source},
        }

        # Check for error columns
        if all(
            col in df.columns for col in ["Susceptible_Error", "Infected_Error", "Recovered_Error"]
        ):
            result["S_err"] = df["Susceptible_Error"].tolist()
            result["I_err"] = df["Infected_Error"].tolist()
            result["R_err"] = df["Recovered_Error"].tolist()
            result["has_error"] = True

        return result

    @classmethod
    def format_result_for_cli(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results for CLI display.

        Args:
            result: Dictionary with results

        Returns:
            Dictionary formatted for display
        """
        formatted: Dict[str, Any] = {}

        # Copy metadata
        if "metadata" in result:
            formatted["metadata"] = result["metadata"]

        # Format simulation data
        if "S_val" in result and "I_val" in result and "R_val" in result:
            formatted["max_infected"] = max(result["I_val"]) if result["I_val"] else 0
            formatted["final_recovered"] = result["R_val"][-1] if result["R_val"] else 0
            formatted["data_points"] = len(result["S_val"])

        # Add error info
        formatted["has_error_data"] = "S_err" in result and "I_err" in result and "R_err" in result

        return formatted

    @classmethod
    def get_metadata_from_path(cls, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file path.

        Args:
            file_path: Path to the results file

        Returns:
            Dictionary with extracted metadata
        """
        metadata: Dict[str, Any] = {}

        try:
            parts = Path(file_path).parts

            # Find the "runs" directory index for new path structure
            # or "spkmc" for legacy structure
            base_idx = -1
            for i, part in enumerate(parts):
                if part == "runs":
                    base_idx = i
                    break
                if part == "spkmc":  # Legacy path support
                    base_idx = i
                    break

            if base_idx >= 0 and len(parts) >= base_idx + 3:
                metadata["distribution"] = parts[base_idx + 1]
                metadata["network"] = parts[base_idx + 2]

                # Extract info from the filename
                filename = Path(file_path).stem
                if filename.startswith("results_"):
                    params = filename.replace("results_", "").split("_")

                    # New format: results_<N>_<samples>_k<k_avg>_exp<exponent>_...
                    # Legacy format: results_<exponent>_<N>_<samples> (for SF only)
                    if len(params) >= 2:
                        try:
                            # Detect format by checking for prefixed params (k, exp, sh, etc.)
                            has_prefixed_params = any(
                                p.startswith(("k", "exp", "sh", "sc", "mu", "lam")) and len(p) > 3
                                for p in params[2:]
                            )

                            # Legacy SF format detection:
                            # - Network is SF
                            # - No prefixed params
                            # - First param is small (likely exponent like 25 for 2.5)
                            # - Has at least 3 params (exponent, N, samples)
                            network = metadata.get("network", "").lower()
                            first_val = int(params[0])
                            is_legacy_sf = (
                                network == "sf"
                                and not has_prefixed_params
                                and len(params) >= 3
                                and first_val < 100  # Exponents are typically < 10
                            )

                            if is_legacy_sf:
                                # Legacy SF format: exponent, N, samples
                                exp_str = params[0]
                                if exp_str.isdigit() and len(exp_str) >= 2:
                                    metadata["exponent"] = float(exp_str[:-1] + "." + exp_str[-1])
                                else:
                                    metadata["exponent"] = float(exp_str)
                                metadata["N"] = int(params[1])
                                metadata["samples"] = int(params[2])
                            else:
                                # New format: N, samples, then key-value params
                                metadata["N"] = first_val
                                metadata["samples"] = int(params[1])

                                # Parse additional params like k10, exp2p5, sh2p0, etc.
                                # Helper to decode param values
                                # (supports both 'p' encoding and legacy)
                                def decode_param(s: str) -> float:
                                    """Decode parameter value from filename.

                                    Handles:
                                    - New format with 'p' for decimal: 2p75 -> 2.75, 0p05 -> 0.05
                                    - Legacy format (single decimal): 25 -> 2.5 (only for digits)
                                    """
                                    if "p" in s:
                                        # New format: replace 'p' with '.'
                                        return float(s.replace("p", "."))
                                    elif s.isdigit() and len(s) >= 2:
                                        # Legacy format: insert decimal before last digit
                                        return float(s[:-1] + "." + s[-1])
                                    else:
                                        return float(s)

                                for param in params[2:]:
                                    if param.startswith("k") and param[1:].isdigit():
                                        metadata["k_avg"] = int(param[1:])
                                    elif param.startswith("exp"):
                                        metadata["exponent"] = decode_param(param[3:])
                                    elif param.startswith("sh"):
                                        metadata["shape"] = decode_param(param[2:])
                                    elif param.startswith("sc"):
                                        metadata["scale"] = decode_param(param[2:])
                                    elif param.startswith("mu"):
                                        metadata["mu"] = decode_param(param[2:])
                                    elif param.startswith("lam"):
                                        metadata["lambda"] = decode_param(param[3:])
                        except (ValueError, IndexError):
                            pass
        except (ValueError, IndexError, AttributeError):
            pass

        return metadata
