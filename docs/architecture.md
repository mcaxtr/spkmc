# Arquitetura do SPKMC

Este documento descreve a arquitetura do pacote SPKMC, explicando os principais componentes e suas interações.

## Visão Geral

O SPKMC (Shortest Path Kinetic Monte Carlo) é um algoritmo para simulação de propagação de epidemias em redes, utilizando o modelo SIR (Susceptible-Infected-Recovered). A implementação é baseada em classes e interfaces que permitem a simulação em diferentes tipos de redes e com diferentes distribuições de probabilidade.

## Estrutura do Projeto

```
spkmc/
├── cli/                # Módulo da CLI
│   ├── __init__.py
│   ├── __main__.py     # Ponto de entrada para python -m spkmc.cli
│   ├── commands.py     # Comandos da CLI
│   └── validators.py   # Validadores para parâmetros da CLI
├── core/               # Módulo principal
│   ├── __init__.py
│   ├── distributions.py # Classes de distribuição
│   ├── networks.py     # Classes de redes
│   └── simulation.py   # Classe SPKMC
├── io/                 # Módulo de entrada/saída
│   ├── __init__.py
│   └── results.py      # Gerenciamento de resultados
├── utils/              # Utilitários
│   ├── __init__.py
│   └── numba_utils.py  # Funções otimizadas com Numba
└── visualization/      # Módulo de visualização
    ├── __init__.py
    └── plots.py        # Funções de visualização
```

## Componentes Principais

### Módulo `core`

O módulo `core` contém as classes principais do algoritmo SPKMC.

#### `distributions.py`

Este arquivo contém as classes de distribuição de probabilidade usadas no SPKMC:

- `Distribution`: Classe abstrata que define a interface para as distribuições.
- `GammaDistribution`: Implementação da distribuição Gamma.
- `ExponentialDistribution`: Implementação da distribuição Exponencial.
- `create_distribution`: Função para criar uma instância de distribuição com base no tipo e parâmetros.

#### `networks.py`

Este arquivo contém as classes para criação e manipulação de redes:

- `NetworkFactory`: Classe para criar diferentes tipos de redes (Erdos-Renyi, Complexa, Grafo Completo).

#### `simulation.py`

Este arquivo contém a implementação principal do algoritmo SPKMC:

- `SPKMC`: Classe principal que implementa o algoritmo SPKMC.

### Módulo `utils`

O módulo `utils` contém funções auxiliares para o algoritmo SPKMC.

#### `numba_utils.py`

Este arquivo contém funções otimizadas com Numba para melhorar o desempenho das simulações:

- `gamma_sampling`: Amostragem da distribuição Gamma.
- `get_weight_exponential`: Amostragem da distribuição Exponencial.
- `compute_infection_times_gamma`: Cálculo dos tempos de infecção usando a distribuição Gamma.
- `compute_infection_times_exponential`: Cálculo dos tempos de infecção usando a distribuição Exponencial.
- `get_states`: Cálculo dos estados (S, I, R) para cada nó em um determinado tempo.
- `calculate`: Cálculo da proporção de indivíduos em cada estado para cada passo de tempo.

### Módulo `io`

O módulo `io` contém classes para gerenciamento de entrada e saída de dados.

#### `results.py`

Este arquivo contém classes para gerenciamento de resultados de simulações:

- `ResultManager`: Classe para salvar e carregar resultados de simulações.

### Módulo `visualization`

O módulo `visualization` contém classes para visualização de resultados.

#### `plots.py`

Este arquivo contém classes para visualização de resultados de simulações:

- `Visualizer`: Classe para visualização de resultados.

### Módulo `cli`

O módulo `cli` contém a implementação da interface de linha de comando.

#### `commands.py`

Este arquivo contém os comandos da CLI:

- `cli`: Grupo principal de comandos.
- `run`: Comando para executar uma simulação.
- `plot`: Comando para visualizar resultados.
- `info`: Comando para mostrar informações sobre simulações salvas.
- `compare`: Comando para comparar resultados de múltiplas simulações.

#### `validators.py`

Este arquivo contém validadores para os parâmetros da CLI:

- `validate_percentage`: Validador para garantir que o valor seja uma porcentagem válida.
- `validate_positive`: Validador para garantir que o valor seja positivo.
- `validate_positive_int`: Validador para garantir que o valor seja um inteiro positivo.
- `validate_network_type`: Validador para garantir que o tipo de rede seja válido.
- `validate_distribution_type`: Validador para garantir que o tipo de distribuição seja válido.

## Fluxo de Execução

1. O usuário cria uma instância de distribuição (`GammaDistribution` ou `ExponentialDistribution`).
2. O usuário cria uma instância de `SPKMC` com a distribuição.
3. O usuário cria uma rede usando `NetworkFactory`.
4. O usuário executa a simulação usando os métodos da classe `SPKMC`.
5. O usuário visualiza os resultados usando `Visualizer`.
6. O usuário salva os resultados usando `ResultManager`.

## Otimizações

O SPKMC utiliza a biblioteca Numba para otimizar as funções críticas do algoritmo. As funções otimizadas estão no arquivo `numba_utils.py` e são decoradas com `@njit` ou `@njit(parallel=True)` para paralelização.

## Interface de Linha de Comando

A CLI do SPKMC permite executar simulações, visualizar resultados e obter informações diretamente do terminal. A CLI é implementada usando a biblioteca Click e está no módulo `cli`.

## Extensibilidade

O SPKMC é projetado para ser extensível. Novas distribuições podem ser adicionadas implementando a classe abstrata `Distribution`. Novos tipos de redes podem ser adicionados implementando novos métodos na classe `NetworkFactory`.