# Fuzzification_v2

## Gabriel Teixeira Júlio

O objetivo deste projeto foi a implementação de um sistema fuzzy de Takagi-Sugeno de ordem zero ou primeira ordem para aproximar a seguinte função não linear: $f(x)=e^{-\frac{x}{5}}\cdot sin(3x)+0.5\cdot sin(x)$. A aproximação está dentro do intervalo $x\in [0,10]$.

## Descrição da Implementação

Para implementar o Takagi-Sugeno foi utilizado o código base encontrado em: [Fuzzification](https://github.com/Kingdrasill/Fuzzification). Para este projeto foi utilizado os arquivos `functions.py` e `domain.py`, e foi criado para este projeto o arquivo `tsk_gradient_descent.py` que está o código para fazer o Takagi-Sugeno e a otimização dele por Gradiente Descendentes implementados em código e o uso do Gradiente Descendente BFGS da biblioteca **_Scipy_**.

Para os testes para aproximação foi implementado dois tipos de Takagi-Sugeno o de zero ordem e primeira ordem. Foi utilizado apenas uma variável linguística chamada de **_X_**, que pode cobrir o intervalo de $x$ com funções de pertinência do tipo: triangular, trapezoidal, gaussiana, sino, cauchy e laplace. Para avaliar o desempenho do sistema fuzzy foi uso o RMSE (Root Mean Square Error) para comparar com o resultado real e depois é usado o Gradiente Descendente para otimizar os valores dos parâmetros do Takagi-Sugeno.

Para gerar o resultado aproximado do Takagi-Sugeno foi criada a função `Gerar_TSK` que retorna o resultado da aproximação. A função recebe os seguintes valores:

- `x`: intervalo de valores de $[0,10]$ que seram calculados
- `entrada`: instância da classe `Domain` que é domínio da variável de entrada
- `paramas`: os parâmetros das regras do Takagi-Sugeno
- `tipo`: defini se as regras tem 1 ou 2 parâmetros em cada

```
# Gera o resultado do método de Takagi-Sugeno de 0 ordem ou 1 ordem
def Gerar_TSK(x, entrada, params, tipo):
    # Guarda o resultado da aproximação
    y = []
    # Para cada valor de x
    for v in x:
        # Calcula os pesos da variável de entrada X
        ws = entrada.calcular_pesos(v)
        # Cada função de pertinência vai ter sua própia regra
        # Se o tipo é de 0 ordem
        if tipo == 0:
            # Para cada regra calcula: ai0
            zs = [(params[i][0]) for i in range(len(params))]
        # Se o tipo é 1 ordem
        elif tipo == 1:
            # Para cada regra calcula: ai0*x + ai1
            zs = [(params[i][0] * v + params[i][1]) for i in range(len(params))]
        # Se não for um tipo implementado
        else:
            return "Tipo de Takagi-Sugeno não existe!"
        # Calcula o numerador: Somatorio dos pesos * as regras
        numerador = sum([ws[i] * zs[i] for i in range(len(params))])
        # Calcula o denominador: Somatorio dos pesos
        denominador = sum(ws)
        # Calcula o TSK: numerador / denominador
        y.append(numerador / denominador)
    return y
```

## Descrição dos Testes
