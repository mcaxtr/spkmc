"""
Validadores para parâmetros da CLI do SPKMC.

Este módulo contém funções de validação para os parâmetros da interface de linha de comando,
garantindo que os valores fornecidos pelo usuário sejam válidos e adequados para a simulação.
"""

import click
from typing import Any, Optional, Callable


def validate_percentage(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validador para garantir que o valor seja uma porcentagem válida (entre 0 e 1).
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o valor não for uma porcentagem válida
    """
    if value < 0 or value > 1:
        raise click.BadParameter("A porcentagem deve estar entre 0 e 1.")
    return value


def validate_positive(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validador para garantir que o valor seja positivo.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o valor não for positivo
    """
    if value <= 0:
        raise click.BadParameter(f"O parâmetro {param.name} deve ser positivo.")
    return value


def validate_positive_int(ctx: click.Context, param: click.Parameter, value: int) -> int:
    """
    Validador para garantir que o valor seja um inteiro positivo.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o valor não for um inteiro positivo
    """
    if not isinstance(value, int) or value <= 0:
        raise click.BadParameter(f"O parâmetro {param.name} deve ser um inteiro positivo.")
    return value


def validate_network_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validador para garantir que o tipo de rede seja válido.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o tipo de rede não for válido
    """
    valid_types = ["er", "cn", "cg"]
    if value.lower() not in valid_types:
        raise click.BadParameter(f"Tipo de rede inválido. Escolha entre: {', '.join(valid_types)}")
    return value.lower()


def validate_distribution_type(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validador para garantir que o tipo de distribuição seja válido.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o tipo de distribuição não for válido
    """
    valid_types = ["gamma", "exponential"]
    if value.lower() not in valid_types:
        raise click.BadParameter(f"Tipo de distribuição inválido. Escolha entre: {', '.join(valid_types)}")
    return value.lower()


def validate_exponent(ctx: click.Context, param: click.Parameter, value: float) -> float:
    """
    Validador para garantir que o expoente da lei de potência seja válido.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o expoente não for válido
    """
    if value <= 1:
        raise click.BadParameter("O expoente da lei de potência deve ser maior que 1.")
    return value


def validate_file_exists(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validador para garantir que o arquivo exista.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o arquivo não existir
    """
    import os
    if value and not os.path.exists(value):
        raise click.BadParameter(f"O arquivo '{value}' não existe.")
    return value


def validate_directory_exists(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validador para garantir que o diretório exista.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o diretório não existir
    """
    import os
    if value and not os.path.isdir(value):
        raise click.BadParameter(f"O diretório '{value}' não existe.")
    return value


def validate_output_file(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """
    Validador para garantir que o arquivo de saída possa ser criado.
    
    Args:
        ctx: Contexto do Click
        param: Parâmetro sendo validado
        value: Valor a ser validado
    
    Returns:
        Valor validado
    
    Raises:
        click.BadParameter: Se o arquivo não puder ser criado
    """
    if not value:
        return value
    
    import os
    try:
        # Verifica se o diretório existe
        directory = os.path.dirname(value)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Verifica se o arquivo pode ser criado
        with open(value, 'a') as f:
            pass
        
        return value
    except Exception as e:
        raise click.BadParameter(f"Não foi possível criar o arquivo de saída: {e}")


def validate_conditional(condition_param: str, condition_value: Any, 
                        validator: Callable[[click.Context, click.Parameter, Any], Any]) -> Callable:
    """
    Cria um validador condicional que só é aplicado se outro parâmetro tiver um valor específico.
    
    Args:
        condition_param: Nome do parâmetro de condição
        condition_value: Valor que o parâmetro de condição deve ter
        validator: Função de validação a ser aplicada
    
    Returns:
        Função de validação condicional
    """
    def conditional_validator(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
        if ctx.params.get(condition_param) == condition_value:
            return validator(ctx, param, value)
        return value
    
    return conditional_validator