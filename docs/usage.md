# Guia de Uso do SPKMC

Este documento fornece informações detalhadas sobre como usar o pacote SPKMC para simulações de propagação de epidemias em redes complexas.

## Instalação

### Requisitos

- Python 3.8 ou superior
- Dependências listadas em `requirements.txt`:
  - NumPy: Operações numéricas eficientes
  - SciPy: Algoritmos científicos e matemáticos
  - NetworkX: Criação e manipulação de redes complexas
  - Matplotlib: Visualização de resultados
  - Numba: Aceleração de código Python
  - tqdm: Barras de progresso
  - Click: Interface de linha de comando
  - Pandas: Manipulação e exportação de dados
  - Colorama: Coloração da saída no terminal
  - Rich: Formatação avançada da saída no terminal
  - openpyxl: Exportação para Excel
  - joblib: Paralelização de tarefas

### Instalação via pip

```bash
# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote em modo de desenvolvimento
pip install -e .
```

### Dependências Opcionais

Algumas funcionalidades do SPKMC dependem de pacotes específicos:

- **Exportação de dados**: Requer `pandas` e `openpyxl` para exportação para CSV e Excel
- **Interface de linha de comando avançada**: Requer `colorama` e `rich` para formatação e coloração da saída
- **Paralelização**: Requer `joblib` para execução paralela de simulações

Se alguma dessas dependências não estiver disponível, o SPKMC continuará funcionando, mas com funcionalidades reduzidas. Mensagens de aviso serão exibidas quando uma funcionalidade não estiver disponível devido à falta de uma dependência.

### Solução de Problemas de Dependências

Se você encontrar erros relacionados a dependências ausentes, como `ModuleNotFoundError: No module named 'pandas'`, certifique-se de que todas as dependências estão instaladas:

```bash
# Verificar se todas as dependências estão instaladas
pip install -r requirements.txt

# Ou instalar uma dependência específica
pip install pandas
```

## Interface de Linha de Comando (CLI)

A CLI do SPKMC fornece uma interface completa para executar simulações, visualizar resultados e obter informações sobre simulações anteriores.

### Opções Globais

As seguintes opções estão disponíveis para todos os comandos da CLI:

| Opção | Descrição |
|-------|-----------|
| `--verbose`, `-v` | Ativar modo verboso para depuração |
| `--no-color` | Desativar cores na saída |
| `--simple` | Gerar arquivo de resultado simplificado em CSV (tempo, infectados, erro). Pode ser usado de duas formas:
|           | - Como opção global antes do comando: `spkmc --simple batch ...`
|           | - Como opção específica após o comando: `spkmc batch ... --simple`
|           | Ambas as formas têm o mesmo efeito. |

### Comandos Disponíveis

A CLI do SPKMC possui os seguintes comandos principais:

- `run`: Executa uma simulação SPKMC
- `plot`: Visualiza resultados de simulações anteriores
- `info`: Mostra informações sobre simulações salvas
- `compare`: Compara resultados de múltiplas simulações

### Comando `run`

O comando `run` executa uma simulação SPKMC com os parâmetros especificados.

#### Sintaxe

```bash
python spkmc_cli.py run [OPÇÕES]
```

#### Opções

| Opção | Abreviação | Tipo | Padrão | Descrição |
|-------|------------|------|--------|-----------|
| `--network-type` | `-n` | string | `er` | Tipo de rede: Erdos-Renyi (er), Complex Network (cn), Complete Graph (cg) |
| `--dist-type` | `-d` | string | `gamma` | Tipo de distribuição: Gamma ou Exponential |
| `--shape` | | float | `2.0` | Parâmetro de forma para distribuição Gamma |
| `--scale` | | float | `1.0` | Parâmetro de escala para distribuição Gamma |
| `--mu` | | float | `1.0` | Parâmetro mu para distribuição Exponencial (recuperação) |
| `--lambda` | | float | `1.0` | Parâmetro lambda para tempos de infecção |
| `--exponent` | | float | `2.5` | Expoente para redes complexas (CN) |
| `--nodes` | `-N` | int | `1000` | Número de nós na rede |
| `--k-avg` | | float | `10` | Grau médio da rede |
| `--samples` | `-s` | int | `50` | Número de amostras por execução |
| `--num-runs` | `-r` | int | `2` | Número de execuções (para médias) |
| `--initial-perc` | `-i` | float | `0.01` | Porcentagem inicial de infectados |
| `--t-max` | | float | `10.0` | Tempo máximo de simulação |
| `--steps` | | int | `100` | Número de passos de tempo |
| `--output` | `-o` | string | | Caminho para salvar os resultados (opcional) |
| `--no-plot` | | flag | `False` | Não exibir o gráfico dos resultados |
| `--overwrite` | | flag | `False` | Sobrescrever resultados existentes |

#### Exemplos

```bash
# Simulação básica com rede Erdos-Renyi e distribuição Gamma
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# Simulação com rede complexa e distribuição Exponencial
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Simulação com grafo completo, salvando resultados
python spkmc_cli.py run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -o results/cg_gamma.json

# Simulação com geração de arquivo CSV simplificado (como opção global antes do comando)
python spkmc_cli.py --simple run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma_simple.json

# Simulação com geração de arquivo CSV simplificado (como opção específica após o comando)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 -o results/er_gamma_simple.json --simple

# Ambas as formas acima têm o mesmo efeito: gerar um arquivo CSV simplificado com tempo, infectados e erro
```

### Comando `plot`

O comando `plot` visualiza os resultados de uma simulação anterior.

#### Sintaxe

```bash
python spkmc_cli.py plot RESULT_FILE [OPÇÕES]
```

#### Argumentos

| Argumento | Descrição |
|-----------|-----------|
| `RESULT_FILE` | Caminho para o arquivo de resultados |

#### Opções

| Opção | Abreviação | Tipo | Padrão | Descrição |
|-------|------------|------|--------|-----------|
| `--with-error` | `-e` | flag | `False` | Mostrar barras de erro (se disponíveis) |
| `--output` | `-o` | string | | Salvar o gráfico em um arquivo (opcional) |

#### Exemplos

```bash
# Visualizar resultados básicos
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json

# Visualizar com barras de erro
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json --with-error

# Salvar o gráfico em um arquivo
python spkmc_cli.py plot data/spkmc/gamma/ER/results_1000_50_2.0.json -o plots/er_gamma.png
```

### Comando `info`

O comando `info` mostra informações sobre simulações salvas.

#### Sintaxe

```bash
python spkmc_cli.py info [OPÇÕES]
```

#### Opções

| Opção | Abreviação | Tipo | Padrão | Descrição |
|-------|------------|------|--------|-----------|
| `--result-file` | `-f` | string | | Arquivo de resultados específico (opcional) |
| `--list` | `-l` | flag | `False` | Listar todos os arquivos de resultados disponíveis |

#### Exemplos

```bash
# Listar todos os arquivos de resultados disponíveis
python spkmc_cli.py info --list

# Mostrar informações sobre um arquivo específico
python spkmc_cli.py info -f data/spkmc/gamma/ER/results_1000_50_2.0.json
```

### Comando `compare`

O comando `compare` compara os resultados de múltiplas simulações.

#### Sintaxe

```bash
python spkmc_cli.py compare RESULT_FILES... [OPÇÕES]
```

#### Argumentos

| Argumento | Descrição |
|-----------|-----------|
| `RESULT_FILES` | Caminhos para os arquivos de resultados (pelo menos um) |

#### Opções

| Opção | Abreviação | Tipo | Padrão | Descrição |
|-------|------------|------|--------|-----------|
| `--labels` | `-l` | string (múltiplo) | | Rótulos para cada arquivo (opcional) |
| `--output` | `-o` | string | | Salvar o gráfico em um arquivo (opcional) |

#### Exemplos

```bash
# Comparar dois arquivos de resultados
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json

# Comparar com rótulos personalizados
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json -l "Gamma" "Exponencial"

# Salvar o gráfico comparativo
python spkmc_cli.py compare data/spkmc/gamma/ER/results_1000_50_2.0.json data/spkmc/exponential/ER/results_1000_50_.json -o plots/comparison.png
```

## Uso Programático

Além da CLI, o SPKMC pode ser usado programaticamente em seus próprios scripts Python.

### Importação

```python
from spkmc import SPKMC, GammaDistribution, ExponentialDistribution, NetworkFactory
import numpy as np
```

### Criação de Distribuições

#### Distribuição Gamma

```python
# Parâmetros: shape, scale, lambda
gamma_dist = GammaDistribution(shape=2.0, scale=1.0, lmbd=1.0)
```

#### Distribuição Exponencial

```python
# Parâmetros: mu, lambda
exp_dist = ExponentialDistribution(mu=1.0, lmbd=1.0)
```

### Criação de Redes

#### Rede Erdos-Renyi

```python
# Parâmetros: número de nós, grau médio
G = NetworkFactory.create_erdos_renyi(N=1000, k_avg=10)
```

#### Rede Complexa

```python
# Parâmetros: número de nós, expoente, grau médio
G = NetworkFactory.create_complex_network(N=1000, exponent=2.5, k_avg=10)
```

#### Grafo Completo

```python
# Parâmetros: número de nós
G = NetworkFactory.create_complete_graph(N=100)
```

### Execução de Simulações

#### Inicialização do Simulador

```python
# Inicializa o simulador com uma distribuição
simulator = SPKMC(distribution=gamma_dist)
```

#### Configuração dos Parâmetros

```python
# Número de nós
N = 1000

# Grau médio
k_avg = 10

# Número de amostras
samples = 50

# Porcentagem inicial de infectados
initial_perc = 0.01

# Configuração do tempo
t_max = 10.0
steps = 100
time_steps = np.linspace(0, t_max, steps)
```

#### Simulação em Rede Erdos-Renyi

```python
# Executa a simulação
S, I, R, S_err, I_err, R_err = simulator.simulate_erdos_renyi(
    num_runs=2,
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Simulação em Rede Complexa

```python
# Executa a simulação
S, I, R, S_err, I_err, R_err = simulator.simulate_complex_network(
    num_runs=2,
    exponent=2.5,
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Simulação em Grafo Completo

```python
# Executa a simulação
S, I, R = simulator.simulate_complete_graph(
    time_steps=time_steps,
    N=N,
    samples=samples,
    initial_perc=initial_perc
)
```

#### Simulação Genérica

```python
# Executa a simulação com base no tipo de rede
result = simulator.run_simulation(
    network_type="er",  # ou "cn", "cg"
    time_steps=time_steps,
    N=N,
    k_avg=k_avg,
    samples=samples,
    initial_perc=initial_perc,
    num_runs=2,
    overwrite=False
)

# Extrair os resultados
S = result["S_val"]
I = result["I_val"]
R = result["R_val"]
has_error = result.get("has_error", False)

if has_error:
    S_err = result["S_err"]
    I_err = result["I_err"]
    R_err = result["R_err"]
```

### Visualização de Resultados

#### Visualização Básica

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(time_steps, S, 'b-', label='Suscetíveis')
plt.plot(time_steps, I, 'r-', label='Infectados')
plt.plot(time_steps, R, 'g-', label='Recuperados')

plt.xlabel('Tempo')
plt.ylabel('Proporção de Indivíduos')
plt.title('Dinâmica do Modelo SIR ao Longo do Tempo')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Visualização com Barras de Erro

```python
from spkmc.visualization.plots import Visualizer

Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps,
                                 title="Simulação SPKMC com Barras de Erro")
```

#### Visualização de Rede

```python
Visualizer.plot_network(G, title="Visualização da Rede")
```

#### Comparação de Resultados

```python
# Resultados de múltiplas simulações
results = [result1, result2]
labels = ["Simulação 1", "Simulação 2"]

Visualizer.compare_results(results, labels, title="Comparação de Simulações")
```

### Gerenciamento de Resultados

#### Salvamento de Resultados

```python
from spkmc.io.results import ResultManager

# Cria um dicionário com os resultados
result = {
    "S_val": list(S),
    "I_val": list(I),
    "R_val": list(R),
    "time": list(time_steps),
    "metadata": {
        "network_type": "ER",
        "distribution": "gamma",
        "N": N,
        "k_avg": k_avg,
        "samples": samples
    }
}

# Salva os resultados
ResultManager.save_result("results/my_simulation.json", result)
```

#### Carregamento de Resultados

```python
# Carrega os resultados
result = ResultManager.load_result("results/my_simulation.json")

# Extrai os dados
S = np.array(result["S_val"])
I = np.array(result["I_val"])
R = np.array(result["R_val"])
time_steps = np.array(result["time"])
```

#### Listagem de Resultados

```python
# Lista todos os resultados disponíveis
result_files = ResultManager.list_results()

# Lista resultados com filtro
gamma_results = ResultManager.list_results(filter_by={"metadata.distribution": "gamma"})
```

#### Extração de Metadados

```python
# Extrai metadados do caminho do arquivo
metadata = ResultManager.get_metadata_from_path("data/spkmc/gamma/ER/results_1000_50_2.0.json")

# Formata o resultado para exibição na CLI
formatted = ResultManager.format_result_for_cli(result)
```

## Exemplos Completos

Para exemplos completos de uso, consulte os seguintes arquivos:

- [Exemplos de uso da CLI](../examples/cli_examples.md)
- [Exemplo básico de simulação](../examples/basic_example.py)
- [Exemplo de uso programático da CLI](../examples/basic_simulation.py)

## Dicas e Boas Práticas

### Escolha do Tipo de Rede

- **Erdos-Renyi (ER)**: Bom para simulações iniciais e testes, pois é uma rede aleatória simples.
- **Rede Complexa (CN)**: Mais realista para modelar redes sociais e biológicas, pois segue uma distribuição de lei de potência.
- **Grafo Completo (CG)**: Útil para casos extremos onde todos os nós estão conectados entre si.

### Escolha da Distribuição

- **Gamma**: Oferece mais flexibilidade na modelagem dos tempos de recuperação, permitindo ajustar a forma e a escala.
- **Exponencial**: Mais simples, assume que os eventos ocorrem a uma taxa constante.

### Otimização de Desempenho

- Para redes grandes (N > 10000), considere reduzir o número de amostras e execuções.
- O parâmetro `--k-avg` (grau médio) afeta significativamente o tempo de execução; valores maiores resultam em mais conexões e cálculos mais demorados.
- Utilize o parâmetro `--no-plot` para evitar a geração de gráficos durante a execução, o que pode economizar tempo e recursos.

### Análise de Resultados

- Compare diferentes tipos de redes com os mesmos parâmetros para entender o impacto da estrutura da rede na propagação.
- Varie os parâmetros da distribuição para observar como os tempos de recuperação e infecção afetam a dinâmica da epidemia.
- Utilize o comando `compare` para visualizar múltiplas simulações em um único gráfico.

### Comando `batch`

O comando `batch` permite executar múltiplos cenários de simulação a partir de um arquivo JSON, facilitando a execução de experimentos em lote.

#### Sintaxe

```bash
python spkmc_cli.py batch [ARQUIVO_CENÁRIOS] [OPÇÕES]
```

#### Argumentos

Argumento | Descrição |
|-----------|-----------|
`ARQUIVO_CENÁRIOS` | Caminho para o arquivo JSON contendo os cenários (padrão: batches.json) |

#### Opções

Opção | Abreviação | Tipo | Padrão | Descrição |
|-------|------------|------|--------|-----------|
`--output-dir` | `-o` | string | `./results` | Diretório para salvar os resultados |
`--prefix` | `-p` | string | | Prefixo para os nomes dos arquivos de saída |
`--compare` | `-c` | flag | `False` | Gerar visualização comparativa dos resultados |
`--no-plot` | | flag | `False` | Desativar a geração de gráficos individuais |
`--save-plot` | | flag | `False` | Salvar os gráficos em arquivos |
`--verbose` | `-v` | flag | `False` | Mostrar informações detalhadas durante a execução |

#### Formato do Arquivo JSON de Cenários

O arquivo JSON deve conter uma lista de objetos, cada um representando um cenário com parâmetros para a simulação. Cada cenário será executado sequencialmente e os resultados serão salvos em arquivos separados no diretório especificado.

Exemplo de arquivo JSON:

```json
[
  {
    "network_type": "er",
    "dist_type": "gamma",
    "nodes": 1000,
    "k_avg": 10,
    "shape": 2.0,
    "scale": 1.0,
    "lambda_val": 0.5,
    "samples": 50,
    "num_runs": 2,
    "initial_perc": 0.01,
    "t_max": 10.0,
    "steps": 100,
    "label": "er_gamma_shape2"
  },
  {
    "network_type": "cn",
    "dist_type": "exponential",
    "nodes": 2000,
    "k_avg": 8,
    "exponent": 2.5,
    "mu": 1.0,
    "lambda_val": 0.5,
    "samples": 50,
    "num_runs": 2,
    "initial_perc": 0.01,
    "t_max": 10.0,
    "steps": 100,
    "label": "cn_exp_exponent2.5"
  }
]
```

Parâmetros disponíveis para cada cenário:

Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
`network_type` | string | `er` | Tipo de rede: Erdos-Renyi (er), Complex Network (cn), Complete Graph (cg) |
`dist_type` | string | `gamma` | Tipo de distribuição: Gamma ou Exponential |
`nodes` | int | `1000` | Número de nós na rede |
`k_avg` | float | `10` | Grau médio da rede (para er e cn) |
`shape` | float | `2.0` | Parâmetro de forma para distribuição Gamma |
`scale` | float | `1.0` | Parâmetro de escala para distribuição Gamma |
`mu` | float | `1.0` | Parâmetro mu para distribuição Exponencial |
`lambda_val` | float | `1.0` | Parâmetro lambda para tempos de infecção |
`exponent` | float | `2.5` | Expoente para redes complexas (CN) |
`samples` | int | `50` | Número de amostras por execução |
`num_runs` | int | `2` | Número de execuções (para médias) |
`initial_perc` | float | `0.01` | Porcentagem inicial de infectados |
`t_max` | float | `10.0` | Tempo máximo de simulação |
`steps` | int | `100` | Número de passos de tempo |
`label` | string | | Rótulo opcional para identificar o cenário (usado no nome do arquivo) |

#### Exemplos

```bash
# Executar cenários do arquivo padrão (batches.json)
python spkmc_cli.py batch

# Executar cenários de um arquivo específico
python spkmc_cli.py batch experimentos/cenarios_teste.json

# Executar cenários e salvar resultados em um diretório específico
python spkmc_cli.py batch --output-dir resultados/experimento1

# Executar cenários com um prefixo para os arquivos de saída
python spkmc_cli.py batch --prefix "exp1_"

# Executar cenários e gerar uma visualização comparativa
python spkmc_cli.py batch --compare

# Executar cenários, salvar gráficos e não exibi-los na tela
python spkmc_cli.py batch --save-plot --no-plot

# Executar cenários com informações detalhadas durante a execução
python spkmc_cli.py batch --verbose

# Executar cenários e gerar arquivos CSV simplificados (como opção global antes do comando)
python spkmc_cli.py --simple batch --output-dir resultados/csv_simples

# Executar cenários e gerar arquivos CSV simplificados (como opção específica após o comando)
python spkmc_cli.py batch --output-dir resultados/csv_simples --simple

# Ambas as formas acima têm o mesmo efeito: gerar arquivos CSV simplificados com tempo, infectados e erro para cada cenário
```

#### Dicas e Boas Práticas para o Comando `batch`

- **Organização de cenários**: Agrupe cenários relacionados em um mesmo arquivo JSON para facilitar a comparação e análise.
- **Uso de rótulos**: Utilize o parâmetro `label` para identificar claramente cada cenário nos arquivos de saída.
- **Comparação automática**: Use a opção `--compare` para gerar automaticamente um gráfico comparativo de todos os cenários executados.
- **Execuções em lote**: Para experimentos extensos, considere dividir os cenários em múltiplos arquivos JSON e executá-los separadamente.
- **Prefixos significativos**: Utilize prefixos que indiquem o propósito do experimento (ex: `gamma_vs_exp_`, `network_size_test_`).
- **Modo silencioso**: Para execuções longas, use `--no-plot` para evitar a exibição de gráficos durante a execução, melhorando o desempenho.
- **Documentação de experimentos**: Salve os arquivos JSON de cenários junto com os resultados para documentar completamente o experimento.
- **Arquivos CSV simplificados**: Use a opção `--simple` para gerar arquivos CSV simplificados com apenas três colunas (tempo, infectados, erro). Esta opção pode ser usada de duas formas:
  - Como opção global antes do comando: `spkmc --simple batch ...`
  - Como opção específica após o comando: `spkmc batch ... --simple`
  Ambas as formas têm o mesmo efeito. Estes arquivos são úteis para análises rápidas ou importação em outras ferramentas de visualização.