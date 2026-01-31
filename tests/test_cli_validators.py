"""
Tests for SPKMC CLI validators.

This module contains tests for validation functions in the SPKMC CLI.
"""

import os
import tempfile

import click
import pytest

from spkmc.cli.validators import (
    validate_conditional,
    validate_directory_exists,
    validate_distribution_type,
    validate_exponent,
    validate_file_exists,
    validate_network_type,
    validate_output_file,
    validate_percentage,
    validate_positive,
    validate_positive_int,
)


class MockContext:
    """Mock for the Click context."""

    def __init__(self):
        self.params = {}


class MockParameter:
    """Mock for the Click parameter."""

    def __init__(self, name):
        self.name = name


@pytest.fixture
def ctx():
    """Fixture for the Click context."""
    return MockContext()


@pytest.fixture
def param():
    """Fixture for the Click parameter."""
    return MockParameter("test_param")


@pytest.fixture
def temp_file():
    """Fixture to create a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    yield path

    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    path = tempfile.mkdtemp()

    yield path

    if os.path.exists(path):
        os.rmdir(path)


def test_validate_percentage_valid(ctx, param):
    """Test percentage validation with valid values."""
    assert validate_percentage(ctx, param, 0.0) == 0.0
    assert validate_percentage(ctx, param, 0.5) == 0.5
    assert validate_percentage(ctx, param, 1.0) == 1.0


def test_validate_percentage_invalid(ctx, param):
    """Test percentage validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_percentage(ctx, param, -0.1)

    with pytest.raises(click.BadParameter):
        validate_percentage(ctx, param, 1.1)


def test_validate_positive_valid(ctx, param):
    """Test positive value validation with valid values."""
    assert validate_positive(ctx, param, 0.1) == 0.1
    assert validate_positive(ctx, param, 1.0) == 1.0
    assert validate_positive(ctx, param, 100.0) == 100.0


def test_validate_positive_invalid(ctx, param):
    """Test positive value validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_positive(ctx, param, 0.0)

    with pytest.raises(click.BadParameter):
        validate_positive(ctx, param, -1.0)


def test_validate_positive_int_valid(ctx, param):
    """Test positive integer validation with valid values."""
    assert validate_positive_int(ctx, param, 1) == 1
    assert validate_positive_int(ctx, param, 10) == 10
    assert validate_positive_int(ctx, param, 100) == 100


def test_validate_positive_int_invalid(ctx, param):
    """Test positive integer validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, 0)

    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, -1)

    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, 1.5)


def test_validate_network_type_valid(ctx, param):
    """Test network type validation with valid values."""
    assert validate_network_type(ctx, param, "er") == "er"
    assert validate_network_type(ctx, param, "ER") == "er"
    assert validate_network_type(ctx, param, "cn") == "cn"
    assert validate_network_type(ctx, param, "cg") == "cg"


def test_validate_network_type_invalid(ctx, param):
    """Test network type validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_network_type(ctx, param, "invalid")

    with pytest.raises(click.BadParameter):
        validate_network_type(ctx, param, "")


def test_validate_distribution_type_valid(ctx, param):
    """Test distribution type validation with valid values."""
    assert validate_distribution_type(ctx, param, "gamma") == "gamma"
    assert validate_distribution_type(ctx, param, "GAMMA") == "gamma"
    assert validate_distribution_type(ctx, param, "exponential") == "exponential"


def test_validate_distribution_type_invalid(ctx, param):
    """Test distribution type validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_distribution_type(ctx, param, "invalid")

    with pytest.raises(click.BadParameter):
        validate_distribution_type(ctx, param, "")


def test_validate_exponent_valid(ctx, param):
    """Test exponent validation with valid values."""
    assert validate_exponent(ctx, param, 1.1) == 1.1
    assert validate_exponent(ctx, param, 2.0) == 2.0
    assert validate_exponent(ctx, param, 3.5) == 3.5


def test_validate_exponent_invalid(ctx, param):
    """Test exponent validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 0.5)

    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 1.0)

    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 0.0)


def test_validate_file_exists_valid(ctx, param, temp_file):
    """Test file existence validation with valid values."""
    assert validate_file_exists(ctx, param, temp_file) == temp_file


def test_validate_file_exists_invalid(ctx, param):
    """Test file existence validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_file_exists(ctx, param, "nonexistent_file.txt")


def test_validate_directory_exists_valid(ctx, param, temp_dir):
    """Test directory existence validation with valid values."""
    assert validate_directory_exists(ctx, param, temp_dir) == temp_dir


def test_validate_directory_exists_invalid(ctx, param):
    """Test directory existence validation with invalid values."""
    with pytest.raises(click.BadParameter):
        validate_directory_exists(ctx, param, "nonexistent_directory")


def test_validate_output_file_valid(ctx, param, temp_dir):
    """Test output file validation with valid values."""
    output_path = os.path.join(temp_dir, "output.txt")
    assert validate_output_file(ctx, param, output_path) == output_path

    # Verify the file was created and remove it
    assert os.path.exists(output_path)
    os.remove(output_path)


def test_validate_output_file_none(ctx, param):
    """Test output file validation with None."""
    assert validate_output_file(ctx, param, None) is None


def test_validate_conditional(ctx, param):
    """Test conditional validation."""

    # Create a simple validator function
    def validate_test(ctx, param, value):
        if value < 0:
            raise click.BadParameter("Value must be non-negative")
        return value

    # Create the conditional validator
    conditional_validator = validate_conditional(
        "condition_param", "condition_value", validate_test
    )

    # Test when the condition is satisfied
    ctx.params["condition_param"] = "condition_value"
    assert conditional_validator(ctx, param, 10) == 10

    with pytest.raises(click.BadParameter):
        conditional_validator(ctx, param, -10)

    # Test when the condition is not satisfied
    ctx.params["condition_param"] = "other_value"
    assert conditional_validator(ctx, param, -10) == -10  # Should not validate
