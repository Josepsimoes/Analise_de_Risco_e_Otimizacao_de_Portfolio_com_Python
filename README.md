# Análise e Otimização de Portfólio de Investimento

Este projeto visa fornecer uma análise abrangente e ferramentas de otimização para portfólios de investimento. Ele aborda uma variedade de tópicos, desde cálculos básicos de retorno até modelos avançados de otimização de portfólio, todos utilizando Python e bibliotecas populares do ecossistema financeiro.

## Funcionalidades

- **Calcular Retornos Diários e Logarítmicos**: Funções para calcular os retornos diários e logarítmicos dos ativos financeiros.
  
- **Visualização em Histograma**: Funções para visualizar a distribuição dos retornos financeiros em forma de histograma.

- **Calcular Desvio-Padrão / Volatilidade**: Funções para calcular o desvio-padrão e volatilidade diária e anual dos retornos financeiros.

- **Drawdown e Downside Risk**: Funções para calcular o drawdown e o downside risk dos retornos financeiros.

- **Semivariância**: Funções para calcular a semivariância dos retornos financeiros.

- **Modelo CAPM e Beta**: Implementação do Modelo de Avaliação de Ativos Financeiros (CAPM) e cálculo do beta.

- **Value at Risk (VaR)**: Cálculo do Valor em Risco para diferentes níveis de confiança.

- **Simulação de Monte Carlo (VaR)**: Implementação de simulação de Monte Carlo para estimar o VaR.

- **Modelos de Otimização de Portfólio**:
  - Mínima-Volatilidade
  - Maximização do Índice de Sharpe
  - Risco Eficiente
  - Retorno Eficiente
  - Com Restrições Setoriais e por Ativo específico

## Utilização

Para utilizar este projeto, basta importar as funções necessárias e chamá-las conforme necessário.

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from functools import reduce
import statsmodels.api as sm
from scipy.stats import norm
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions
```

## Requisitos

- Python 3.x
- Bibliotecas: pandas, numpy, yfinance, matplotlib, plotly, statsmodels, scipy, pypfopt

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue para relatar bugs ou sugerir novas funcionalidades. Pull requests também são encorajados.
