# SPKMC - Shortest Path Kinetic Monte Carlo

Uma implementação eficiente do algoritmo Shortest Path Kinetic Monte Carlo (SPKMC) para simulação de propagação de epidemias em redes complexas, utilizando o modelo SIR (Susceptible-Infected-Recovered).

## Descrição

O SPKMC é um algoritmo estocástico para simular a propagação de epidemias em redes complexas. Esta implementação permite a simulação em diferentes tipos de redes (Erdos-Renyi, Redes Complexas com distribuição de lei de potência, Grafos Completos) e com diferentes distribuições de probabilidade (Gamma, Exponencial) para os tempos de recuperação e infecção.

O algoritmo SPKMC utiliza o conceito de caminhos mais curtos para calcular os tempos de propagação da epidemia entre os nós da rede, o que o torna mais eficiente computacionalmente do que métodos tradicionais de simulação de Monte Carlo para redes grandes.

### Características Principais

- Simulação de epidemias em diferentes tipos de redes
- Suporte a diferentes distribuições de probabilidade
- Cálculo eficiente de caminhos mais curtos usando algoritmos otimizados
- Interface de linha de comando (CLI) completa
- Visualização de resultados com gráficos informativos
- Comparação entre diferentes simulações
- Exportação de resultados em formato JSON

## Instalação

### Requisitos

- Python 3.8 ou superior
- Conhecimentos básicos de linha de comando

### Passo a Passo para Instalação

A instalação do SPKMC segue o fluxo padrão de instalação de pacotes Python. Siga os passos abaixo:

#### 1. Instalar o virtualenv

O virtualenv é uma ferramenta para criar ambientes Python isolados. Isso é importante para evitar conflitos entre dependências de diferentes projetos e é recomendado, já que atualmente é difícil instalar pacotes Python usando pip sem estar em um ambiente virtual.

**Opção 1: Instalação via pip**
```bash
pip install virtualenv
```

**Opção 2: Instalação via gerenciadores de pacotes do sistema operacional**

Em muitos sistemas, é recomendável instalar o virtualenv através do gerenciador de pacotes do sistema operacional, o que pode resolver problemas de permissão e dependências:

Para Ubuntu/Debian:
```bash
sudo apt-get install python3-virtualenv
```

Para Fedora:
```bash
sudo dnf install python3-virtualenv
```

Para macOS (usando Homebrew):
```bash
brew install virtualenv
```

Para Windows (usando Chocolatey):
```bash
choco install virtualenv
```

#### 2. Criar um ambiente virtual

Crie um ambiente virtual para isolar as dependências do SPKMC:

```bash
# No Windows
virtualenv venv

# No Linux/macOS
python3 -m virtualenv venv
```

#### 3. Ativar o ambiente virtual

Ative o ambiente virtual criado:

```bash
# No Windows
venv\Scripts\activate

# No Linux/macOS
source venv/bin/activate
```

Após a ativação, você verá o nome do ambiente virtual no início da linha de comando, como `(venv)`.

#### 4. Clonar o repositório (se ainda não tiver feito)

```bash
git clone https://github.com/seu-usuario/spkmc.git
cd spkmc
```

#### 5. Instalar o pacote em modo de desenvolvimento

Instale o pacote em modo de desenvolvimento para que as alterações no código sejam refletidas automaticamente:

```bash
pip install -e .
```

Este comando instalará o SPKMC e todas as suas dependências automaticamente. Após a instalação, o comando `spkmc` estará disponível no seu ambiente virtual.

## Uso Básico

Após a instalação, você terá acesso ao comando `spkmc` que fornece uma interface de linha de comando completa para executar simulações, visualizar resultados e obter informações.

### Interface de Linha de Comando (CLI)

#### Executar uma simulação

```bash
spkmc run --network-type er --dist-type gamma --nodes 1000 --k-avg 10 --samples 50
```

#### Visualizar resultados

```bash
spkmc plot data/spkmc/gamma/ER/results_1000_50_2.0.json
```

#### Obter informações sobre simulações salvas

```bash
spkmc info --result-file data/spkmc/gamma/ER/results_1000_50_2.0.json
```

#### Listar simulações disponíveis

```bash
spkmc info --list
```

#### Comparar resultados de múltiplas simulações

```bash
spkmc compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json
```

### Uso Programático

Você também pode usar o SPKMC programaticamente em seus próprios scripts Python:

```python
import numpy as np
from spkmc import SPKMC, GammaDistribution, NetworkFactory

# Criar distribuição
gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)

# Criar simulador
simulator = SPKMC(gamma_dist)

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
spkmc run -n er -d gamma --shape 2.0 --scale 1.0 --lambda 1.0 -N 1000 --k-avg 10 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

### Simulação com Rede Complexa e Distribuição Exponencial

```bash
spkmc run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

### Simulação com Grafo Completo

```bash
spkmc run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -i 0.01 --t-max 10.0 --steps 100
```

### Execução em Lote

Você pode executar múltiplos cenários de simulação a partir de um arquivo JSON:

```bash
spkmc batch --scenarios-file batches.json --output-dir results/batch_test
```

Para exemplos mais detalhados, consulte o diretório `examples/`:
- [Exemplos de uso da CLI](examples/cli_examples.md)
- [Exemplo básico de simulação](examples/basic_example.py)
- [Exemplo de uso programático da CLI](examples/basic_simulation.py)

## Parâmetros Comuns

### Tipos de Rede
- `er`: Erdos-Renyi (rede aleatória)
- `cn`: Complex Network (rede complexa com distribuição de lei de potência)
- `cg`: Complete Graph (grafo completo)

### Tipos de Distribuição
- `gamma`: Distribuição Gamma
- `exponential`: Distribuição Exponencial

### Parâmetros de Simulação
- `-N, --nodes`: Número de nós na rede
- `--k-avg`: Grau médio da rede (para redes ER e CN)
- `-s, --samples`: Número de amostras por execução
- `-i, --initial-perc`: Porcentagem inicial de infectados
- `--t-max`: Tempo máximo de simulação
- `--steps`: Número de passos de tempo

## Solução de Problemas (Troubleshooting)

### Problemas de Instalação

#### Erro: "Command 'pip' not found"
- **Solução**: Certifique-se de que o Python está instalado corretamente e adicionado ao PATH do sistema. Você também pode tentar usar `python -m pip` em vez de apenas `pip`.

#### Erro ao instalar dependências
- **Solução**: Verifique se você está usando uma versão compatível do Python (3.8 ou superior). Algumas dependências podem requerer compiladores específicos. No Windows, pode ser necessário instalar o Visual C++ Build Tools.

```bash
pip install --upgrade pip setuptools wheel
```

#### Ambiente virtual não ativa corretamente
- **Solução**: Verifique se você está usando o comando correto para o seu sistema operacional. No Windows, use `venv\Scripts\activate`, e no Linux/macOS, use `source venv/bin/activate`.

#### Problemas de permissão ao instalar pacotes
- **Solução**: Se você encontrar problemas de permissão ao instalar pacotes, considere usar o gerenciador de pacotes do seu sistema operacional para instalar o virtualenv, como mostrado na seção de instalação.

### Problemas de Execução

#### Comando 'spkmc' não encontrado
- **Solução**: Certifique-se de que o ambiente virtual está ativado e que o pacote foi instalado corretamente com `pip install -e .`. Você pode verificar se o comando está disponível com:

```bash
pip list | grep spkmc
```

#### Erro: "ModuleNotFoundError"
- **Solução**: Este erro geralmente ocorre quando uma dependência não está instalada. Certifique-se de que você instalou o pacote com `pip install -e .`, que instalará todas as dependências necessárias.

#### Erro de memória durante simulações grandes
- **Solução**: Reduza o tamanho da rede ou o número de amostras. Para redes muito grandes, considere usar uma máquina com mais memória RAM.

#### Visualização não mostra gráficos
- **Solução**: Certifique-se de que o matplotlib está instalado corretamente. Em alguns ambientes, pode ser necessário configurar um backend específico:

```python
import matplotlib
matplotlib.use('TkAgg')  # ou outro backend compatível
```

#### Erro ao salvar resultados
- **Solução**: Verifique se você tem permissões de escrita no diretório de destino. Use o parâmetro `--output` para especificar um caminho alternativo.

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
│   └── numba_utils.py  # Funções otimizadas com Numba
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