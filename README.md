# SPKMC - Shortest Path Kinetic Monte Carlo

Uma implementação eficiente do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC) para simulação de propagação de epidemias em redes complexas, utilizando o modelo SIR (Susceptible-Infected-Recovered).

## Descrição

O SPKMC é um algoritmo estocástico para simular a propagação de epidemias em redes complexas. Esta implementação permite a simulação em diferentes tipos de redes (Erdos-Renyi, Redes Complexas com distribuição de lei de potência, Grafos Completos) e com diferentes distribuições de probabilidade (Gamma, Exponencial) para os tempos de recuperação e infecção.

O algoritmo SPKMC utiliza o conceito de caminhos mais curtos para calcular os tempos de propagação da epidemia entre os nós da rede, o que o torna mais eficiente computacionalmente do que métodos tradicionais de simulação de Monte Carlo para redes grandes.

### Características Principais

- Simulação de epidemias em diferentes tipos de redes
- Suporte a diferentes distribuições de probabilidade
- Cálculo eficiente de caminhos mais curtos usando algoritmos otimizados
- Aceleração GPU para simulações de alta performance
- Interface de linha de comando (CLI) completa
- Visualização de resultados com gráficos informativos
- Comparação entre diferentes simulações
- Exportação de resultados em formato JSON

## Instalação

### Requisitos

- Python 3.8 ou superior
- Dependências:
  - NumPy: Operações numéricas eficientes
  - SciPy: Algoritmos científicos e matemáticos
  - NetworkX: Criação e manipulação de redes complexas
  - Matplotlib: Visualização de resultados
  - Numba: Aceleração de código Python
  - tqdm: Barras de progresso
  - Click: Interface de linha de comando

### Instalação via pip

```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote em modo de desenvolvimento
pip install -e .
```

### Instalação a partir do código-fonte

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/spkmc.git
cd spkmc

# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote
pip install -e .
```

### Instalação com Suporte a GPU

Para instalar o SPKMC com suporte a aceleração GPU:

```bash
# Instalar o pacote com suporte a GPU
pip install -e ".[gpu]"
```

Isso instalará as dependências adicionais necessárias para aceleração GPU (cupy-cuda12x, cudf-cu12, cugraph-cu12). Note que essas dependências requerem uma GPU NVIDIA compatível com CUDA 12.x.

## Uso Básico

### Interface de Linha de Comando (CLI)

O pacote fornece uma CLI completa para executar simulações, visualizar resultados e obter informações.

#### Executar uma simulação

```bash
python -m spkmc.cli run --network-type er --dist-type gamma --nodes 1000 --k-avg 10 --samples 50
```

ou usando o script principal:

```bash
python spkmc_cli.py run --network-type er --dist-type gamma --nodes 1000 --k-avg 10 --samples 50
```

Para usar a aceleração GPU:

```bash
python spkmc_cli.py --gpu run --network-type er --dist-type gamma --nodes 1000 --k-avg 10 --samples 50
```

#### Visualizar resultados

```bash
python -m spkmc.cli plot data/spkmc/gamma/ER/results_1000_50_2.0.json
```

#### Obter informações sobre simulações salvas

```bash
python -m spkmc.cli info --result-file data/spkmc/gamma/ER/results_1000_50_2.0.json
```

#### Listar simulações disponíveis

```bash
python -m spkmc.cli info --list
```

#### Comparar resultados de múltiplas simulações

```bash
python -m spkmc.cli compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json
```

### Uso Programático

Você também pode usar o SPKMC programaticamente em seus próprios scripts Python:

```python
import numpy as np
from spkmc import SPKMC, GammaDistribution, NetworkFactory

# Criar distribuição
gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)

# Criar simulador (sem GPU)
simulator = SPKMC(gamma_dist)

# Ou com aceleração GPU
# simulator = SPKMC(gamma_dist, use_gpu=True)

# Configurar parâmetros
N = 1000
k_avg = 10
samples = 50
time_steps = np.linspace(0, 10.0, 100)

# Criar rede
G = NetworkFactory.create_erdos_renyi(N, k_avg)

# Configurar nós inicialmente infectados
sources = np.random.randint(0, N, int(N * 0.01))

# Executar simulação
S, I, R = simulator.run_multiple_simulations(G, sources, time_steps, samples)

# Visualizar resultados
import matplotlib.pyplot as plt
plt.plot(time_steps, S, 'b-', label='Suscetíveis')
plt.plot(time_steps, I, 'r-', label='Infectados')
plt.plot(time_steps, R, 'g-', label='Recuperados')
plt.legend()
plt.show()
```

## Exemplos

### Simulação com Rede Erdos-Renyi e Distribuição Gamma

```bash
python -m spkmc.cli run -n er -d gamma --shape 2.0 --scale 1.0 --lambda 1.0 -N 1000 --k-avg 10 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

### Simulação com Rede Complexa e Distribuição Exponencial

```bash
python -m spkmc.cli run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

### Simulação com Grafo Completo

```bash
python -m spkmc.cli run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -i 0.01 --t-max 10.0 --steps 100

# Simulação com aceleração GPU para redes grandes
python -m spkmc.cli --gpu run -n er -d gamma --shape 2.0 --scale 1.0 -N 5000 --k-avg 10 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

Para exemplos mais detalhados, consulte o diretório `examples/`:
- [Exemplos de uso da CLI](examples/cli_examples.md)
- [Exemplo básico de simulação](examples/basic_example.py)
- [Exemplo de uso programático da CLI](examples/basic_simulation.py)

## Documentação

A documentação completa está disponível nos seguintes arquivos:

- [Guia de Uso](docs/usage.md): Documentação detalhada sobre como usar o SPKMC
- [Arquitetura](docs/architecture.md): Descrição da arquitetura do projeto

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
│   ├── numba_utils.py  # Funções otimizadas com Numba
│   └── gpu_utils.py    # Funções para aceleração GPU
└── visualization/      # Módulo de visualização
    ├── __init__.py
    └── plots.py        # Funções de visualização
```

## Contribuição

Contribuições são bem-vindas! Se você deseja contribuir com este projeto, siga estas etapas:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Faça commit das suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Faça push para a branch (`git push origin feature/nova-feature`)
5. Crie um novo Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Citação

Se você usar este software em sua pesquisa, por favor cite:

```
@software{spkmc2025,
  author = {SPKMC Team},
  title = {SPKMC: Shortest Path Kinetic Monte Carlo},
  year = {2025},
  url = {https://github.com/seu-usuario/spkmc}
}
```

## Contato

Para questões, sugestões ou problemas, por favor abra uma issue no repositório do projeto.