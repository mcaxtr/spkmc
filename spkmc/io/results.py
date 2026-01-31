"""
Result management for the SPKMC algorithm.

This module contains functions and classes to manage simulation results,
including saving, loading, and data handling.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from spkmc.core.distributions import Distribution


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

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


class ResultManager:
    """Manage saving and loading of simulation results."""

    # Base directory for results
    BASE_DIR = "data/spkmc"

    @staticmethod
    def get_result_path(
        network_type: str,
        distribution: Distribution,
        N: int,
        samples: int,
        exponent: Optional[float] = None,
        k_avg: Optional[float] = None,
    ) -> str:
        """
        Generate the path for the results file.

        Args:
            network_type: Network type (ER, CN, CG, RRN)
            distribution: Distribution object
            N: Number of nodes
            samples: Number of samples
            exponent: Exponent for complex networks
            k_avg: Average degree

        Returns:
            Path to the results file
        """
        base_path = (
            f"{ResultManager.BASE_DIR}/{distribution.get_distribution_name()}/{network_type}/"
        )

        # Create directories if they do not exist
        Path(base_path).mkdir(parents=True, exist_ok=True)

        # Build filename with all relevant parameters
        k_avg_str = f"_k{int(k_avg)}" if k_avg is not None else ""

        if network_type.upper() == "CN" and exponent is not None:
            exp_str = str(exponent).replace(".", "")
            params_str = distribution.get_params_string()
            return f"{base_path}results_{exp_str}_{N}_{samples}{k_avg_str}_{params_str}.json"
        else:
            params_str = distribution.get_params_string()
            return f"{base_path}results_{N}_{samples}{k_avg_str}_{params_str}.json"

    @staticmethod
    def load_result(file_path: str) -> Dict[str, Any]:
        """
        Load results from a JSON file.

        Args:
            file_path: Path to the results file

        Returns:
            Dictionary with results

        Raises:
            FileNotFoundError: If the file does not exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        try:
            with open(file_path, "r") as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise e

    @staticmethod
    def save_result(file_path: str, result: Dict[str, Any]) -> None:
        """
        Save results to a JSON file.

        Args:
            file_path: Path to the results file
            result: Dictionary with results

        Raises:
            IOError: If an error occurs while saving the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyJSONEncoder)
        except Exception as e:
            raise IOError(f"Error saving results: {e}")

    @staticmethod
    def list_results(filter_by: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List available results files.

        Args:
            filter_by: Dictionary with filter criteria (optional)

        Returns:
            List of paths to results files
        """
        results: List[str] = []
        base_path = Path(ResultManager.BASE_DIR)

        if not base_path.exists():
            return results

        for dist_dir in base_path.iterdir():
            if dist_dir.is_dir():
                for network_dir in dist_dir.iterdir():
                    if network_dir.is_dir():
                        for result_file in network_dir.glob("*.json"):
                            # If there is no filter, add all results
                            if filter_by is None:
                                results.append(str(result_file))
                            else:
                                # Check whether the file meets filter criteria
                                try:
                                    result_data = ResultManager.load_result(str(result_file))

                                    # Check whether all filter criteria are met
                                    match = True
                                    for key, value in filter_by.items():
                                        if key not in result_data or result_data[key] != value:
                                            match = False
                                            break

                                    if match:
                                        results.append(str(result_file))
                                except (json.JSONDecodeError, OSError, KeyError):
                                    # Ignore files that cannot be loaded
                                    pass

        return results

    @staticmethod
    def get_metadata_from_path(file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file path.

        Args:
            file_path: Path to the results file

        Returns:
            Dictionary with extracted metadata
        """
        metadata: Dict[str, Any] = {}

        try:
            # Extract path info
            parts = Path(file_path).parts

            # Find the base directory index
            base_idx = -1
            for i, part in enumerate(parts):
                if part == "spkmc":
                    base_idx = i
                    break

            if base_idx >= 0 and len(parts) >= base_idx + 3:
                metadata["distribution"] = parts[base_idx + 1]
                metadata["network_type"] = parts[base_idx + 2]

                # Extract info from the filename
                filename = Path(file_path).stem
                if filename.startswith("results_"):
                    params = filename.replace("results_", "").split("_")

                    if metadata["network_type"].lower() == "cn" and len(params) >= 3:
                        # For complex networks: exponent, N, samples, [params]
                        # Exponent is stored without decimal: "25" means "2.5"
                        try:
                            exp_str = params[0]
                            if exp_str.isdigit() and len(exp_str) >= 2:
                                # Insert decimal before last digit: "25" -> "2.5"
                                metadata["exponent"] = float(exp_str[:-1] + "." + exp_str[-1])
                            else:
                                metadata["exponent"] = float(exp_str)
                            metadata["N"] = int(params[1])
                            metadata["samples"] = int(params[2])
                        except Exception:
                            pass
                    elif len(params) >= 2:
                        # For other networks: N, samples, [params]
                        try:
                            metadata["N"] = int(params[0])
                            metadata["samples"] = int(params[1])
                        except (ValueError, IndexError):
                            pass
        except (ValueError, IndexError, AttributeError):
            pass

        return metadata

    @staticmethod
    def format_result_for_cli(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format results for CLI display.

        Args:
            result: Dictionary with results

        Returns:
            Dictionary formatted for display
        """
        formatted = {}

        # Copy metadata
        if "metadata" in result:
            formatted["metadata"] = result["metadata"]

        # Format simulation data
        if "S_val" in result and "I_val" in result and "R_val" in result:
            formatted["max_infected"] = max(result["I_val"]) if result["I_val"] else 0
            formatted["final_recovered"] = result["R_val"][-1] if result["R_val"] else 0
            formatted["data_points"] = len(result["S_val"])

        # Add error info
        if "S_err" in result and "I_err" in result and "R_err" in result:
            formatted["has_error_data"] = True
        else:
            formatted["has_error_data"] = False

        return formatted

    @staticmethod
    def load_results_from_directory(
        directory: Union[str, Path],
    ) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        Load all JSON results files from a directory.

        Args:
            directory: Path to the directory

        Returns:
            List of tuples (file_path, result_data)

        Raises:
            ValueError: If the directory does not exist or contains no JSON files
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        json_files = list(dir_path.glob("*.json"))

        if not json_files:
            raise ValueError(f"No JSON files found in directory: {directory}")

        results = []

        for json_file in sorted(json_files):
            try:
                data = ResultManager.load_result(str(json_file))
                results.append((json_file, data))
            except Exception as e:
                # Log error but keep processing other files
                print(f"Warning: Error loading {json_file}: {e}")
                continue

        if not results:
            raise ValueError(f"No valid JSON files found in directory: {directory}")

        return results
