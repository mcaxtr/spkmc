"""
Exportação de resultados para o algoritmo SPKMC.

Este módulo contém funções para exportação de resultados de simulações em diferentes
formatos, como CSV, Excel, JSON, Markdown e HTML.
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Importações condicionais
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Aviso: Pandas não encontrado. A exportação para Excel/CSV não estará disponível.")

from spkmc.visualization.plots import Visualizer


class ExportManager:
    """Gerencia a exportação de resultados em diferentes formatos."""
    
    @staticmethod
    def export_to_csv(result: Dict[str, Any], output_path: str) -> str:
        """
        Exporta os resultados para um arquivo CSV.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            
        Returns:
            Caminho para o arquivo exportado
            
        Raises:
            ImportError: Se o pandas não estiver disponível
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas não está instalado. Instale com 'pip install pandas' para exportar para CSV.")
            
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Extrai os dados
        time_steps = np.array(result.get("time", []))
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        
        # Cria o DataFrame
        data = {
            "Time": time_steps,
            "Susceptible": S,
            "Infected": I,
            "Recovered": R
        }
        
        # Adiciona dados de erro, se disponíveis
        if "S_err" in result and "I_err" in result and "R_err" in result:
            data["Susceptible_Error"] = np.array(result.get("S_err", []))
            data["Infected_Error"] = np.array(result.get("I_err", []))
            data["Recovered_Error"] = np.array(result.get("R_err", []))
        
        df = pd.DataFrame(data)
        
        # Salva como CSV
        df.to_csv(output_path, index=False)
        
        return output_path
    
    @staticmethod
    def export_to_excel(result: Dict[str, Any], output_path: str) -> str:
        """
        Exporta os resultados para um arquivo Excel.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            
        Returns:
            Caminho para o arquivo exportado
            
        Raises:
            ImportError: Se o pandas ou openpyxl não estiverem disponíveis
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas não está instalado. Instale com 'pip install pandas openpyxl' para exportar para Excel.")
            
        # Verifica se openpyxl está disponível
        try:
            import openpyxl
        except ImportError:
            raise ImportError("Openpyxl não está instalado. Instale com 'pip install openpyxl' para exportar para Excel.")
            
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Extrai os dados
        time_steps = np.array(result.get("time", []))
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        
        # Cria o DataFrame para os dados
        data = {
            "Time": time_steps,
            "Susceptible": S,
            "Infected": I,
            "Recovered": R
        }
        
        # Adiciona dados de erro, se disponíveis
        if "S_err" in result and "I_err" in result and "R_err" in result:
            data["Susceptible_Error"] = np.array(result.get("S_err", []))
            data["Infected_Error"] = np.array(result.get("I_err", []))
            data["Recovered_Error"] = np.array(result.get("R_err", []))
        
        df_data = pd.DataFrame(data)
        
        # Cria o DataFrame para os metadados
        metadata = result.get("metadata", {})
        df_metadata = pd.DataFrame(list(metadata.items()), columns=["Parameter", "Value"])
        
        # Cria o arquivo Excel com múltiplas planilhas
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_data.to_excel(writer, sheet_name='Data', index=False)
            df_metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Adiciona uma planilha com estatísticas
            stats = {
                "Statistic": ["Max Infected", "Final Recovered", "Time to Peak"],
                "Value": [
                    np.max(I),
                    R[-1] if len(R) > 0 else 0,
                    time_steps[np.argmax(I)] if len(I) > 0 else 0
                ]
            }
            pd.DataFrame(stats).to_excel(writer, sheet_name='Statistics', index=False)
        
        return output_path
    
    @staticmethod
    def export_to_json(result: Dict[str, Any], output_path: str) -> str:
        """
        Exporta os resultados para um arquivo JSON.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            
        Returns:
            Caminho para o arquivo exportado
        """
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Salva como JSON
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return output_path
    
    @staticmethod
    def export_to_markdown(result: Dict[str, Any], output_path: str, include_plot: bool = True) -> str:
        """
        Exporta os resultados para um arquivo Markdown.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            include_plot: Se True, inclui um gráfico no relatório
            
        Returns:
            Caminho para o arquivo exportado
        """
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Extrai os metadados
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")
        
        # Extrai os dados
        time_steps = np.array(result.get("time", []))
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        
        # Calcula estatísticas
        max_infected = np.max(I) if len(I) > 0 else 0
        max_infected_time = time_steps[np.argmax(I)] if len(I) > 0 and len(time_steps) > 0 else 0
        final_recovered = R[-1] if len(R) > 0 else 0
        
        # Gera o conteúdo Markdown
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md_content = f"""# Relatório de Simulação SPKMC

Gerado em: {timestamp}

## Parâmetros da Simulação

| Parâmetro | Valor |
|-----------|-------|
| Tipo de Rede | {network_type} |
| Distribuição | {dist_type} |
| Número de Nós (N) | {N} |
"""
        
        # Adiciona parâmetros específicos
        for key, value in metadata.items():
            if key not in ["network_type", "distribution", "N"]:
                md_content += f"| {key} | {value} |\n"
        
        # Adiciona estatísticas
        md_content += f"""
## Estatísticas

| Estatística | Valor |
|-------------|-------|
| Máximo de Infectados | {max_infected:.4f} |
| Tempo do Pico de Infecção | {max_infected_time:.4f} |
| Recuperados Finais | {final_recovered:.4f} |

"""
        
        # Adiciona gráfico, se solicitado
        if include_plot:
            plot_path = output_path.replace(".md", ".png")
            
            # Gera o gráfico
            title = f"Simulação SPKMC - Rede {network_type}, Distribuição {dist_type}, N={N}"
            
            has_error = "S_err" in result and "I_err" in result and "R_err" in result
            if has_error:
                S_err = np.array(result.get("S_err", []))
                I_err = np.array(result.get("I_err", []))
                R_err = np.array(result.get("R_err", []))
                Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, plot_path)
            else:
                Visualizer.plot_result(S, I, R, time_steps, title, plot_path)
            
            # Adiciona referência ao gráfico no Markdown
            md_content += f"""
## Visualização

![Gráfico da Simulação]({os.path.basename(plot_path)})

"""
        
        # Adiciona tabela de dados (primeiros e últimos 5 pontos)
        md_content += """
## Dados da Simulação

### Primeiros 5 pontos

| Tempo | Suscetíveis | Infectados | Recuperados |
|-------|-------------|------------|-------------|
"""
        
        for i in range(min(5, len(time_steps))):
            md_content += f"| {time_steps[i]:.4f} | {S[i]:.4f} | {I[i]:.4f} | {R[i]:.4f} |\n"
        
        md_content += """
### Últimos 5 pontos

| Tempo | Suscetíveis | Infectados | Recuperados |
|-------|-------------|------------|-------------|
"""
        
        for i in range(max(0, len(time_steps) - 5), len(time_steps)):
            md_content += f"| {time_steps[i]:.4f} | {S[i]:.4f} | {I[i]:.4f} | {R[i]:.4f} |\n"
        
        # Salva o arquivo Markdown
        with open(output_path, 'w') as f:
            f.write(md_content)
        
        return output_path
    
    @staticmethod
    def export_to_html(result: Dict[str, Any], output_path: str, include_plot: bool = True) -> str:
        """
        Exporta os resultados para um arquivo HTML.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            include_plot: Se True, inclui um gráfico no relatório
            
        Returns:
            Caminho para o arquivo exportado
            
        Raises:
            ImportError: Se o pandas não estiver disponível
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas não está instalado. Instale com 'pip install pandas' para exportar para HTML.")
            
        # Primeiro exporta para Markdown
        md_path = output_path.replace(".html", ".md")
        ExportManager.export_to_markdown(result, md_path, include_plot)
        
        # Converte o Markdown para HTML usando pandas
        with open(md_path, 'r') as f:
            md_content = f.read()
        
        # Cria um DataFrame com o conteúdo Markdown
        df = pd.DataFrame({"markdown": [md_content]})
        
        # Converte para HTML
        html = df.to_html(escape=False, index=False, header=False)
        
        # Adiciona estilos CSS
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório SPKMC</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>
"""
        
        # Salva o arquivo HTML
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        # Remove o arquivo Markdown temporário
        os.remove(md_path)
        
        return output_path
    
    @staticmethod
    def export_plot(result: Dict[str, Any], output_path: str, format: str = "png", dpi: int = 300) -> str:
        """
        Exporta o gráfico da simulação em diferentes formatos.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            format: Formato do gráfico (png, pdf, svg, jpg)
            dpi: Resolução do gráfico em DPI
            
        Returns:
            Caminho para o arquivo exportado
        """
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Extrai os metadados
        metadata = result.get("metadata", {})
        network_type = metadata.get("network_type", "").upper()
        dist_type = metadata.get("distribution", "").capitalize()
        N = metadata.get("N", "")
        
        # Extrai os dados
        time_steps = np.array(result.get("time", []))
        S = np.array(result.get("S_val", []))
        I = np.array(result.get("I_val", []))
        R = np.array(result.get("R_val", []))
        
        # Gera o gráfico
        title = f"Simulação SPKMC - Rede {network_type}, Distribuição {dist_type}, N={N}"
        
        has_error = "S_err" in result and "I_err" in result and "R_err" in result
        if has_error:
            S_err = np.array(result.get("S_err", []))
            I_err = np.array(result.get("I_err", []))
            R_err = np.array(result.get("R_err", []))
            Visualizer.plot_result_with_error(S, I, R, S_err, I_err, R_err, time_steps, title, output_path)
        else:
            Visualizer.plot_result(S, I, R, time_steps, title, output_path)
        
        return output_path
    
    @staticmethod
    def export_results(result: Dict[str, Any], output_path: str, format: str = "json") -> str:
        """
        Exporta os resultados no formato especificado.
        
        Args:
            result: Dicionário com os resultados
            output_path: Caminho para o arquivo de saída
            format: Formato de exportação (json, csv, excel, md, html)
            
        Returns:
            Caminho para o arquivo exportado
            
        Raises:
            ValueError: Se o formato for inválido
            ImportError: Se as dependências necessárias não estiverem disponíveis
        """
        format = format.lower()
        
        try:
            if format == "json":
                return ExportManager.export_to_json(result, output_path)
            elif format == "csv":
                if not PANDAS_AVAILABLE:
                    raise ImportError("Pandas não está instalado. Instale com 'pip install pandas' para exportar para CSV.")
                return ExportManager.export_to_csv(result, output_path)
            elif format == "excel":
                if not PANDAS_AVAILABLE:
                    raise ImportError("Pandas não está instalado. Instale com 'pip install pandas openpyxl' para exportar para Excel.")
                return ExportManager.export_to_excel(result, output_path)
            elif format == "md" or format == "markdown":
                return ExportManager.export_to_markdown(result, output_path)
            elif format == "html":
                if not PANDAS_AVAILABLE:
                    raise ImportError("Pandas não está instalado. Instale com 'pip install pandas' para exportar para HTML.")
                return ExportManager.export_to_html(result, output_path)
            else:
                raise ValueError(f"Formato de exportação inválido: {format}")
        except ImportError as e:
            print(f"Erro de importação: {e}")
            print(f"Tentando exportar para JSON como alternativa...")
            return ExportManager.export_to_json(result, output_path.replace(f".{format}", ".json"))