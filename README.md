# Fuzzification_v2

## Gabriel Teixeira Júlio

O objetivo deste projeto foi a implementação de um sistema fuzzy de Takagi-Sugeno de ordem zero e primeira ordem para aproximar a seguinte função não linear: $f(x)=e^{-\frac{x}{5}}\cdot sin(3x)+0.5\cdot sin(x)$. A aproximação está dentro do intervalo $x\in [0,10]$.

## Descrição da Implementação

Para implementar o Takagi-Sugeno foi utilizado o código base encontrado em: [Fuzzyfication](https://github.com/Kingdrasill/Fuzzification). Para este projeto foi utilizado os arquivos `functions.py` e `domain.py`, e foi criado para este projeto o arquivo `tsk_gradient_descent.py` que está o código para fazer o Takagi-Sugeno e a otimização dele por Gradiente Descendentes implementados em código e o uso do Gradiente Descendente BFGS da biblioteca **_Scipy_**.



## Descrição dos Testes
