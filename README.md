# Fuzzification_v2

## Gabriel Teixeira Júlio

Este projeto teve como objetivo implementar um sistema fuzzy de Takagi-Sugeno de ordem zero e primeira ordem para aproximar a seguinte função não linear:

$$
f(x)=e^{−\frac{x}{5}} \cdot sin(3x) + 0.5  \cdot  sin(x)
$$

## Descrição da Implementação

Para implementar o Takagi-Sugeno foi utilizado o código base encontrado em: [Fuzzification](https://github.com/Kingdrasill/Fuzzification). Para este projeto foi utilizado os arquivos `functions.py` e `domain.py`, e foi criado para este projeto o arquivo `tsk_gradient_descent.py` que está o código para fazer o Takagi-Sugeno e a otimização dele por Gradiente Descendentes implementados em código e o uso do Gradiente Descendente BFGS da biblioteca **_Scipy_**.

Para os testes para aproximação foi implementado dois tipos de Takagi-Sugeno o de zero ordem e primeira ordem. Foi utilizado apenas uma variável linguística chamada de **_X_**, que pode cobrir o intervalo de $x$ com funções de pertinência do tipo: triangular, trapezoidal, gaussiana, sino, cauchy e laplace. Para avaliar o desempenho do sistema fuzzy foi uso o RMSE (Root Mean Square Error) para comparar com o resultado real e depois é usado o Gradiente Descendente para otimizar os valores dos parâmetros do Takagi-Sugeno.

### Takagi-Sugeno

Para gerar o resultado aproximado do Takagi-Sugeno foi criada a função `Gerar_TSK` que retorna o resultado da aproximação. A função recebe os seguintes valores:

- `x`: intervalo de valores de $[0,10]$ que seram calculados
- `entrada`: instância da classe `Domain` que é domínio da variável de entrada
- `paramas`: os parâmetros das regras do Takagi-Sugeno
- `tipo`: defini se as regras tem 1 ou 2 parâmetros em cada

Para calcular o resultado é pego cada valor do intervalo, calculado os pesos das funções de pertinência para o valor do intervalo, depois é calculado o valor das regras (cada função de pertinência tem sua própia regra) pelo tipo passado, e depois antes de passar para o próximo valor é calculado e salvo o valor de Takagi-Sugeno do valor do intervalo. Isto é feito para todos os valores do intervalo e depois é retornado o resultado.

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

Para poder otimizar os parâmetros das regras do Takaagi-Sugeno foi utilizado o método Gradiente Descendente. O **Gradiente Descendente** é um algoritmo de otimização usado para minimizar uma função, a ideia principal é ajustar iterativamente os parâmetros de um modelo na direção oposta ao gradiente (ou seja, a derivada) da função de erro em relação aos parâmetros, para reduzir o valor dessa função de erro. Foram usados os gradientes descentes:

- **Gradiente Descendete com Momento**: é uma variação do algoritmo de gradiente descendente que visa acelerar a convergência e reduzir as oscilações nas atualizações dos parâmetros. Ele adiciona um termo de "momento" à atualização, que é uma média ponderada das atualizações passadas.
- **Gradiente Descendete com Adam**: é uma técnica de otimização que combina as vantagens do Gradiente Descendente com Momento e do Gradiente Descendente com Taxa de Aprendizado Adaptativa. Ele ajusta a taxa de aprendizado para cada parâmetro com base nas primeiras e segundas estimativas dos momentos (média e variância) dos gradientes.
- **Gradiente Descendete com RMSprop**: é um algoritmo de otimização que adapta a taxa de aprendizado para cada parâmetro com base na média dos quadrados dos gradientes. Ele foi projetado para superar problemas de gradiente muito grande ou muito pequeno em modelos de aprendizado profundo, proporcionando uma atualização mais estável e eficiente.
- **Minimize BFGS**: é um método iterativo de otimização usado para minimizar uma função em várias variáveis. Ele pertence à classe de métodos de quase-Newton, que visam encontrar o mínimo de uma função sem precisar calcular a inversa da matriz Hessiana (a matriz das segundas derivadas) explicitamente, o que o torna mais eficiente do que os métodos de Newton. Este não foi implementado em código, mas utilizado da biblioteca **_Scipy_** usando o método de otimização dela `minimize`.

Para todos os gradientes descendetes implementados foi utilizado como função de erro a ser reduzida o valor RMSE da função aproximada com o função real. Além disso todos métodos tinham no máximo 5000 iterações para convergir, outra maneira de convergir é a taxa de tolerância do erro ser menor do que 0.00001.

## Descrição dos Testes
