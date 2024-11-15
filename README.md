﻿# Fuzzification_v2

## Gabriel Teixeira Júlio

O objetivo deste projeto foi a implementação de um sistema fuzzy de Takagi-Sugeno de ordem zero e primeira ordem para aproximar a seguinte função não linear: $f(x)=e^{-\frac{x}{5}}\cdot sin(3x)+0.5\cdot sin(x)$. A aproximação está dentro do intervalo $x\in [0,10]$.

## Descrição da Implementação

Para implementar o Takagi-Sugeno foi utilizado o código base encontrado em: [Fuzzyfication](https://github.com/Kingdrasill/Fuzzification). Para este projeto foi utilizado os arquivos `functions.py` e `domain.py`, e criado o arquivo `tsk_gradient_descent.py`

## Descrição dos Testes
