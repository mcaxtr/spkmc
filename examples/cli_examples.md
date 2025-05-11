# Exemplos de Uso da CLI do SPKMC

Este documento contém exemplos detalhados de uso da interface de linha de comando (CLI) do SPKMC para diferentes cenários de simulação.

## Instalação

Antes de usar a CLI, instale as dependências:

```bash
pip install -r requirements.txt
```

Ou instale o pacote em modo de desenvolvimento:

```bash
pip install -e .
```

## Simulação Básica com Rede Erdos-Renyi e Distribuição Gamma

A rede Erdos-Renyi é um modelo de rede aleatória onde cada par de nós tem a mesma probabilidade de estar conectado. A distribuição Gamma é frequentemente usada para modelar tempos de recuperação em epidemias.

### Exemplo Básico

```bash
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 --lambda 1.0 -N 1000 --k-avg 10 -s 50
```

Este comando executa uma simulação com:
- Rede Erdos-Renyi (`-n er`)
- Distribuição Gamma (`-d gamma`) com parâmetros de forma (`--shape 2.0`) e escala (`--scale 1.0`)
- 1000 nós (`-N 1000`)
- Grau médio de 10 (`--k-avg 10`)
- 50 amostras por execução (`-s 50`)
- Parâmetro lambda para tempos de infecção (`--lambda 1.0`)

### Variações de Parâmetros

#### Variando o Tamanho da Rede

```bash
# Rede pequena (500 nós)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 500 --k-avg 10 -s 50

# Rede média (2000 nós)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 2000 --k-avg 10 -s 50

# Rede grande (5000 nós)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 5000 --k-avg 10 -s 50
```

#### Variando o Grau Médio

```bash
# Grau médio baixo (5)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 5 -s 50

# Grau médio médio (15)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 15 -s 50

# Grau médio alto (30)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 30 -s 50
```

#### Variando os Parâmetros da Distribuição Gamma

```bash
# Forma baixa (1.0)
python spkmc_cli.py run -n er -d gamma --shape 1.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# Forma alta (3.0)
python spkmc_cli.py run -n er -d gamma --shape 3.0 --scale 1.0 -N 1000 --k-avg 10 -s 50

# Escala baixa (0.5)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 0.5 -N 1000 --k-avg 10 -s 50

# Escala alta (2.0)
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 2.0 -N 1000 --k-avg 10 -s 50
```

### Salvando Resultados

```bash
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/er_gamma.json
```

## Simulação com Rede Complexa e Distribuição Exponencial

As redes complexas seguem uma distribuição de grau de lei de potência, o que as torna mais realistas para modelar muitas redes do mundo real. A distribuição exponencial é frequentemente usada para modelar tempos de eventos aleatórios.

### Exemplo Básico

```bash
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50
```

Este comando executa uma simulação com:
- Rede Complexa (`-n cn`)
- Distribuição Exponencial (`-d exponential`) com parâmetros mu (`--mu 1.0`) e lambda (`--lambda 1.0`)
- Expoente da lei de potência (`--exponent 2.5`)
- 1000 nós (`-N 1000`)
- Grau médio de 10 (`--k-avg 10`)
- 50 amostras por execução (`-s 50`)

### Variações de Parâmetros

#### Variando o Expoente da Lei de Potência

```bash
# Expoente baixo (2.1) - mais hubs, cauda mais longa
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.1 -N 1000 --k-avg 10 -s 50

# Expoente médio (2.5) - típico de muitas redes reais
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Expoente alto (3.0) - menos hubs, mais homogêneo
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 3.0 -N 1000 --k-avg 10 -s 50
```

#### Variando os Parâmetros da Distribuição Exponencial

```bash
# Mu baixo (0.5) - recuperação mais lenta
python spkmc_cli.py run -n cn -d exponential --mu 0.5 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Mu alto (2.0) - recuperação mais rápida
python spkmc_cli.py run -n cn -d exponential --mu 2.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Lambda baixo (0.5) - infecção mais lenta
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 0.5 --exponent 2.5 -N 1000 --k-avg 10 -s 50

# Lambda alto (2.0) - infecção mais rápida
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 2.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50
```

### Salvando Resultados

```bash
python spkmc_cli.py run -n cn -d exponential --mu 1.0 --lambda 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -o results/cn_exponential.json
```

## Comparação entre Diferentes Tipos de Redes

Para comparar diferentes tipos de redes, execute simulações para cada tipo e depois use o comando `compare`.

### Executando Simulações para Diferentes Redes

```bash
# Rede Erdos-Renyi
python spkmc_cli.py run -n er -d gamma --shape 2.0 --scale 1.0 -N 1000 --k-avg 10 -s 50 -o results/er_gamma.json

# Rede Complexa
python spkmc_cli.py run -n cn -d gamma --shape 2.0 --scale 1.0 --exponent 2.5 -N 1000 --k-avg 10 -s 50 -o results/cn_gamma.json

# Grafo Completo
python spkmc_cli.py run -n cg -d gamma --shape 2.0 --scale 1.0 -N 500 -s 50 -o results/cg_gamma.json
```

### Comparando os Resultados

```bash
python spkmc_cli.py compare results/er_gamma.json results/cn_gamma.json results/cg_gamma.json -l "Erdos-Renyi" "Rede Complexa" "Grafo Completo" -o plots/network_comparison.png
```

## Visualização de Resultados Salvos

Após executar simulações e salvar os resultados, você pode visualizá-los posteriormente.

### Visualização Básica

```bash
python spkmc_cli.py plot results/er_gamma.json
```

### Visualização com Barras de Erro

```bash
python spkmc_cli.py plot results/er_gamma.json --with-error
```

### Salvando o Gráfico

```bash
python spkmc_cli.py plot results/er_gamma.json -o plots/er_gamma_plot.png
```

### Obtendo Informações sobre os Resultados

```bash
python spkmc_cli.py info -f results/er_gamma.json
```

## Exportação de Resultados para Diferentes Formatos

Os resultados são salvos em formato JSON por padrão, mas você pode convertê-los para outros formatos usando scripts Python.

### Exemplo de Conversão para CSV

Crie um script Python como este:

```python
import json
import csv
import sys

def json_to_csv(json_file, csv_file):
    # Carregar os dados JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extrair os dados
    time = data.get('time', [])
    S = data.get('S_val', [])
    I = data.get('I_val', [])
    R = data.get('R_val', [])
    
    # Escrever no CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Susceptible', 'Infected', 'Recovered'])
        for t, s, i, r in zip(time, S, I, R):
            writer.writerow([t, s, i, r])
    
    print(f"Dados convertidos de {json_file} para {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python json_to_csv.py <arquivo_json> <arquivo_csv>")
        sys.exit(1)
    
    json_to_csv(sys.argv[1], sys.argv[2])
```

Execute o script:

```bash
python json_to_csv.py results/er_gamma.json results/er_gamma.csv
```

### Exemplo de Conversão para Excel

Crie um script Python como este (requer a biblioteca `pandas` e `openpyxl`):

```python
import json
import pandas as pd
import sys

def json_to_excel(json_file, excel_file):
    # Carregar os dados JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extrair os dados
    time = data.get('time', [])
    S = data.get('S_val', [])
    I = data.get('I_val', [])
    R = data.get('R_val', [])
    
    # Criar DataFrame
    df = pd.DataFrame({
        'Time': time,
        'Susceptible': S,
        'Infected': I,
        'Recovered': R
    })
    
    # Extrair metadados
    metadata = data.get('metadata', {})
    
    # Criar um escritor Excel
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Escrever os dados
        df.to_excel(writer, sheet_name='Data', index=False)
        
        # Escrever os metadados
        pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"Dados convertidos de {json_file} para {excel_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python json_to_excel.py <arquivo_json> <arquivo_excel>")
        sys.exit(1)
    
    json_to_excel(sys.argv[1], sys.argv[2])
```

Execute o script:

```bash
python json_to_excel.py results/er_gamma.json results/er_gamma.xlsx
```

## Ajuda

### Mostrar Ajuda Geral

```bash
python spkmc_cli.py --help
```

### Mostrar Ajuda para um Comando Específico

```bash
python spkmc_cli.py run --help
python spkmc_cli.py plot --help
python spkmc_cli.py info --help
python spkmc_cli.py compare --help