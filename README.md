# Fuzzification_v2

## Gabriel Teixeira Júlio

Este projeto teve como objetivo implementar um sistema fuzzy de Takagi-Sugeno de ordem zero e primeira ordem para aproximar a seguinte função não linear:

$$
f(x)=e^{−\frac{x}{5}} \cdot sin(3x) + 0.5  \cdot  sin(x)
$$

A aproximação foi realizada no intervalo $x \in [0,10]$.

## Descrição da Implementação

Para a implementação do modelo Takagi-Sugeno, utilizou-se como base o código disponibilizado em [Fuzzification](https://github.com/Kingdrasill/Fuzzification). Foram usdados os arquivos `functions.py` e `domain.py`, além da criação do arquivo `tsk_gradient_descent.py`, desenvolvido especificamente para este projeto. Neste arquivo, encontram-se as implementações do modelo Takagi-Sugeno e da otimização de seus parâmetros por Gradiente Descendente. Adicionalmente, utilizou-se também o método minimize usando BFGS da biblioteca **_Scipy_**.

### Takagi-Sugeno

O sistema fuzzy foi testado com dois modelos:

- **Takagi-Sugeno de ordem zero**
- **Takagi-Sugeno de primeira ordem**

Foi utilizada apenas uma variável linguística chamada X, que cobre o intervalo $x$ com funções de pertinência do tipo: triangular, trapezoidal, gaussiana, sino, Cauchy e Laplace. Para avaliar o desempenho do sistema, utilizou-se a métrica RMSE (Root Mean Square Error), que compara os resultados aproximados com os valores reais. Em seguida, aplicou-se o Gradiente Descendente para otimizar os parâmetros das regras do Takagi-Sugeno.

A função `Gerar_TSK` foi implementada para calcular a aproximação do modelo. Ela recebe os seguintes parâmetros:

- `x`: intervalo de valores de $[0,10]$ para os quais será realizada a aproximação.
- `entrada`: instância da classe `Domain` representando o domínio da variável de entrada.
- `paramas`: parâmetros das regras do Takagi-Sugeno.
- `tipo`: define se as regras são de ordem zero ou primeira ordem.

O cálculo é realizado iterativamente para cada valor de $x$, pegando os pesos das funções de pertinência, ponderando os valores das regras, e computando o resultado final da aproximação.

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

### RMSE e Gradiente Descendente

A otimização dos parâmetros foi realizada utilizando o Gradiente Descendente, que minimiza iterativamente o erro entre a função real e a aproximada. Foram implementadas as seguintes variações:

- **Gradiente Descendete com Momento**: Acelera a convergência e reduz oscilações ao incorporar o histórico das atualizações anteriores.
- **Gradiente Descendete com Adam**: Combina momento e adaptação da taxa de aprendizado para cada parâmetro, ajustando-se dinamicamente às características do gradiente.
- **Gradiente Descendete com RMSprop**: Adapta a taxa de aprendizado com base na média dos quadrados dos gradientes, proporcionando atualizações mais estáveis.
- **Minimize BFGS**: Método quasi-Newton que minimiza uma função em várias variáveis. Utilizado diretamente da biblioteca Scipy.

A métrica de erro utilizada em todas as abordagens foi o RMSE, com um limite de tolerância de $10^{-5}$ ou, no máximo, $5000$ iterações para convergência.

## Descrição dos Testes
