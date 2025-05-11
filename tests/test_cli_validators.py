"""
Testes para os validadores da CLI do SPKMC.

Este módulo contém testes para as funções de validação da interface de linha de comando do SPKMC.
"""

import os
import tempfile
import pytest
import click
from click.testing import CliRunner

from spkmc.cli.validators import (
    validate_percentage,
    validate_positive,
    validate_positive_int,
    validate_network_type,
    validate_distribution_type,
    validate_exponent,
    validate_file_exists,
    validate_directory_exists,
    validate_output_file,
    validate_conditional
)


class MockContext:
    """Mock para o contexto do Click."""
    def __init__(self):
        self.params = {}


class MockParameter:
    """Mock para o parâmetro do Click."""
    def __init__(self, name):
        self.name = name


@pytest.fixture
def ctx():
    """Fixture para o contexto do Click."""
    return MockContext()


@pytest.fixture
def param():
    """Fixture para o parâmetro do Click."""
    return MockParameter("test_param")


@pytest.fixture
def temp_file():
    """Fixture para criar um arquivo temporário."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
    
    yield path
    
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_dir():
    """Fixture para criar um diretório temporário."""
    path = tempfile.mkdtemp()
    
    yield path
    
    if os.path.exists(path):
        os.rmdir(path)


def test_validate_percentage_valid(ctx, param):
    """Testa a validação de porcentagem com valores válidos."""
    assert validate_percentage(ctx, param, 0.0) == 0.0
    assert validate_percentage(ctx, param, 0.5) == 0.5
    assert validate_percentage(ctx, param, 1.0) == 1.0


def test_validate_percentage_invalid(ctx, param):
    """Testa a validação de porcentagem com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_percentage(ctx, param, -0.1)
    
    with pytest.raises(click.BadParameter):
        validate_percentage(ctx, param, 1.1)


def test_validate_positive_valid(ctx, param):
    """Testa a validação de valor positivo com valores válidos."""
    assert validate_positive(ctx, param, 0.1) == 0.1
    assert validate_positive(ctx, param, 1.0) == 1.0
    assert validate_positive(ctx, param, 100.0) == 100.0


def test_validate_positive_invalid(ctx, param):
    """Testa a validação de valor positivo com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_positive(ctx, param, 0.0)
    
    with pytest.raises(click.BadParameter):
        validate_positive(ctx, param, -1.0)


def test_validate_positive_int_valid(ctx, param):
    """Testa a validação de inteiro positivo com valores válidos."""
    assert validate_positive_int(ctx, param, 1) == 1
    assert validate_positive_int(ctx, param, 10) == 10
    assert validate_positive_int(ctx, param, 100) == 100


def test_validate_positive_int_invalid(ctx, param):
    """Testa a validação de inteiro positivo com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, 0)
    
    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, -1)
    
    with pytest.raises(click.BadParameter):
        validate_positive_int(ctx, param, 1.5)


def test_validate_network_type_valid(ctx, param):
    """Testa a validação de tipo de rede com valores válidos."""
    assert validate_network_type(ctx, param, "er") == "er"
    assert validate_network_type(ctx, param, "ER") == "er"
    assert validate_network_type(ctx, param, "cn") == "cn"
    assert validate_network_type(ctx, param, "cg") == "cg"


def test_validate_network_type_invalid(ctx, param):
    """Testa a validação de tipo de rede com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_network_type(ctx, param, "invalid")
    
    with pytest.raises(click.BadParameter):
        validate_network_type(ctx, param, "")


def test_validate_distribution_type_valid(ctx, param):
    """Testa a validação de tipo de distribuição com valores válidos."""
    assert validate_distribution_type(ctx, param, "gamma") == "gamma"
    assert validate_distribution_type(ctx, param, "GAMMA") == "gamma"
    assert validate_distribution_type(ctx, param, "exponential") == "exponential"


def test_validate_distribution_type_invalid(ctx, param):
    """Testa a validação de tipo de distribuição com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_distribution_type(ctx, param, "invalid")
    
    with pytest.raises(click.BadParameter):
        validate_distribution_type(ctx, param, "")


def test_validate_exponent_valid(ctx, param):
    """Testa a validação de expoente com valores válidos."""
    assert validate_exponent(ctx, param, 1.1) == 1.1
    assert validate_exponent(ctx, param, 2.0) == 2.0
    assert validate_exponent(ctx, param, 3.5) == 3.5


def test_validate_exponent_invalid(ctx, param):
    """Testa a validação de expoente com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 0.5)
    
    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 1.0)
    
    with pytest.raises(click.BadParameter):
        validate_exponent(ctx, param, 0.0)


def test_validate_file_exists_valid(ctx, param, temp_file):
    """Testa a validação de existência de arquivo com valores válidos."""
    assert validate_file_exists(ctx, param, temp_file) == temp_file


def test_validate_file_exists_invalid(ctx, param):
    """Testa a validação de existência de arquivo com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_file_exists(ctx, param, "nonexistent_file.txt")


def test_validate_directory_exists_valid(ctx, param, temp_dir):
    """Testa a validação de existência de diretório com valores válidos."""
    assert validate_directory_exists(ctx, param, temp_dir) == temp_dir


def test_validate_directory_exists_invalid(ctx, param):
    """Testa a validação de existência de diretório com valores inválidos."""
    with pytest.raises(click.BadParameter):
        validate_directory_exists(ctx, param, "nonexistent_directory")


def test_validate_output_file_valid(ctx, param, temp_dir):
    """Testa a validação de arquivo de saída com valores válidos."""
    output_path = os.path.join(temp_dir, "output.txt")
    assert validate_output_file(ctx, param, output_path) == output_path
    
    # Verifica se o arquivo foi criado e depois o remove
    assert os.path.exists(output_path)
    os.remove(output_path)


def test_validate_output_file_none(ctx, param):
    """Testa a validação de arquivo de saída com None."""
    assert validate_output_file(ctx, param, None) is None


def test_validate_conditional(ctx, param):
    """Testa a validação condicional."""
    # Cria uma função de validação simples
    def validate_test(ctx, param, value):
        if value < 0:
            raise click.BadParameter("Valor deve ser não-negativo")
        return value
    
    # Cria o validador condicional
    conditional_validator = validate_conditional("condition_param", "condition_value", validate_test)
    
    # Testa quando a condição é satisfeita
    ctx.params["condition_param"] = "condition_value"
    assert conditional_validator(ctx, param, 10) == 10
    
    with pytest.raises(click.BadParameter):
        conditional_validator(ctx, param, -10)
    
    # Testa quando a condição não é satisfeita
    ctx.params["condition_param"] = "other_value"
    assert conditional_validator(ctx, param, -10) == -10  # Não deve validar