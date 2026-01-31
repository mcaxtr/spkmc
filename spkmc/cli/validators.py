"""
Validators for SPKMC CLI parameters.

This module contains validation functions for CLI parameters,
ensuring user-provided values are valid and suitable for simulation.
"""

from typing import Any, Callable

import click


def validate_percentage(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validator to ensure the value is a valid percentage (between 0 and 1).

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the value is not a valid percentage
    """
    if value < 0 or value > 1:
        raise click.BadParameter("The percentage must be between 0 and 1.")
    return value


def validate_positive(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validator to ensure the value is positive.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the value is not positive
    """
    if value <= 0:
        raise click.BadParameter(f"Parameter {param.name} must be positive.")
    return value


def validate_positive_int(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """
    Validator to ensure the value is a positive integer.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the value is not a positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise click.BadParameter(f"Parameter {param.name} must be a positive integer.")
    return value


def validate_network_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validator to ensure the network type is valid.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the network type is not valid
    """
    valid_types = ["er", "cn", "cg", "rrn"]
    if value.lower() not in valid_types:
        raise click.BadParameter(f"Invalid network type. Choose from: {', '.join(valid_types)}")
    return value.lower()


def validate_distribution_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validator to ensure the distribution type is valid.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the distribution type is not valid
    """
    valid_types = ["gamma", "exponential"]
    if value.lower() not in valid_types:
        raise click.BadParameter(
            f"Invalid distribution type. Choose from: {', '.join(valid_types)}"
        )
    return value.lower()


def validate_exponent(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validator to ensure the power-law exponent is valid.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the exponent is not valid
    """
    if value <= 1:
        raise click.BadParameter("The power-law exponent must be greater than 1.")
    return value


def validate_file_exists(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validator to ensure the file exists.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the file does not exist
    """
    import os

    if value and not os.path.exists(value):
        raise click.BadParameter(f"The file '{value}' does not exist.")
    return value


def validate_directory_exists(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validator to ensure the directory exists.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the directory does not exist
    """
    import os

    if value and not os.path.isdir(value):
        raise click.BadParameter(f"The directory '{value}' does not exist.")
    return value


def validate_output_file(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validator to ensure the output file can be created.

    Args:
        ctx: Click context
        param: Parameter being validated
        value: Value to validate

    Returns:
        Validated value

    Raises:
        click.BadParameter: If the file cannot be created
    """
    if not value:
        return value

    import os

    try:
        # Check whether the directory exists
        directory = os.path.dirname(value)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Check whether the file can be created
        with open(value, "a"):
            pass

        return value
    except Exception as e:
        raise click.BadParameter(f"Unable to create the output file: {e}")


def validate_conditional(
    condition_param: str,
    condition_value: Any,
    validator: Callable[[click.Context, click.Parameter, Any], Any],
) -> Callable:
    """
    Create a conditional validator applied only when another parameter has a specific value.

    Args:
        condition_param: Name of the condition parameter
        condition_value: Value the condition parameter must have
        validator: Validation function to apply

    Returns:
        Conditional validation function
    """

    def conditional_validator(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        if ctx.params.get(condition_param) == condition_value:
            return validator(ctx, param, value)
        return value

    return conditional_validator
