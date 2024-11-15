from src.domain import *
from scipy.optimize import minimize

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

# Função usada pelo scipy.minimize para otimizar o TSK usando o RMSE
def RMSE_Objective(flat_params, x, fx, entrada, num_params, tipo, errors):
    params = flat_params.reshape(-1, num_params)
    # Calcula o resultado atual do TSK
    aprox = Gerar_TSK(x, entrada, params, tipo)
    # Calcula o RMSE do resultda atual do TSK
    rmse = (np.sqrt(np.square(np.array(aprox) - np.array(fx)))).mean()
    # Guarda o RMSE
    errors.append(rmse)
    return rmse

# Gera o resultado da aproxiação otimizado
def Gerar_Resultado_Aproximado(x, fx, entrada, method, tipo, max_iter):
    # Se o tipo é 0 ordem
    if tipo == 0:
        # Cada regra vai ter 1 parâmetro
        num_params = 1
    # Se o tipo é 1 ordem
    elif tipo == 1:
        # Cada regra vai ter 2 parâmetros
        num_params = 2
    # Se não é um tipo implementado
    else:
        return ("Tipo de Takagi-Sugeno passado não existe!", None)
    
    # Cada função de pertinência vai ter sua própia regra
    # Começa todos os parâmetros das regras em zero
    params = [np.zeros(num_params) for i in range(len(entrada.funcs))]

    # Guarda os erros
    erros = []

    # Ajusta os parâmetros para o método minimize
    initial_params = np.array([param for sublist in params for param in sublist]).flatten()

    # Otimiza a aproximação pelo método RMSE_Objective
    result = minimize(
        RMSE_Objective,                                 # Método objetivo
        initial_params,                                 # Parâmetros iniciais
        args=(x, fx, entrada, num_params, tipo, erros), # Argumentos para o método objetivo
        method=method,                                  # O método de gradiente descendente a ser usado
        tol=1e-5,                                       # Tolerância do erro de 0.00001
        options={'maxiter': max_iter, 'disp': True}         # Número máximo de iterações e mostra o resultado depois
    )

    # Ajuta os parâmetros otimizados
    optimized_params = result.x.reshape(-1, num_params)

    # Retorna o resultado aproximado otimizado e os erros
    aprox_optimized = Gerar_TSK(x, entrada, optimized_params, tipo)
    return (aprox_optimized, erros)

# Função que implementa o gradiente descendente com momento
def Gradient_Descent_Momentum(x, fx, entrada, tipo, learning_rate=1e-4, max_iter=1000, tol=1e-5, momentum=0.9):
    if tipo == 0:
        num_params = 1
    elif tipo == 1:
        num_params = 2
    else:
        raise ValueError("Tipo de Takagi-Sugeno não existe!")

    params = np.zeros((len(entrada.funcs), num_params))  # Inicializa os parâmetros como zeros
    errors = []
    epsilon = 1e-8

    velocity = np.zeros_like(params)  # Inicializa o termo de momento

    for iteration in range(max_iter):
        aprox = Gerar_TSK(x, entrada, params, tipo)  # Gera as aproximações do modelo
        rmse = np.sqrt(np.mean((np.array(aprox) - np.array(fx))**2))  # Calcula o erro RMSE
        errors.append(rmse)  # Armazena o erro da iteração atual

        if iteration > 0 and abs(errors[-1] - errors[-2]) < tol:  # Critério de convergência
            print(f"Converged after {iteration} iterations.")
            break

        gradients = np.zeros_like(params)  # Inicializa os gradientes

        for i, v in enumerate(x):  # Itera sobre os dados de entrada
            ws = entrada.calcular_pesos(v)  # Calcula os pesos fuzzy para o ponto atual
            zs = [params[j][0] * v + params[j][1] if tipo == 1 else params[j][0] for j in range(len(params))]
            numerador = sum(ws[j] * zs[j] for j in range(len(params)))  # Soma ponderada
            denominador = sum(ws) + epsilon  # Evita divisão por zero
            y_hat = numerador / denominador  # Calcula a saída estimada
            error = y_hat - fx[i]  # Calcula o erro

            for j in range(len(params)):  # Calcula o gradiente para cada parâmetro
                dz = [v, 1] if tipo == 1 else [1]
                dnum = ws[j] * np.array(dz)
                dy_dparams = dnum / denominador
                gradients[j] += error * dy_dparams

        velocity = momentum * velocity - learning_rate * gradients  # Atualiza a velocidade com momento
        params += velocity  # Atualiza os parâmetros

    return params, errors

# Função que implementa o gradiente descendente com adam
def Gradient_Descent_Adam(x, fx, entrada, tipo, learning_rate=1e-3, max_iter=1000, tol=1e-5, beta1=0.9, beta2=0.999):
    if tipo == 0:
        num_params = 1
    elif tipo == 1:
        num_params = 2
    else:
        raise ValueError("Tipo de Takagi-Sugeno não existe!")

    params = np.zeros((len(entrada.funcs), num_params))  # Inicializa os parâmetros como zeros
    errors = []
    epsilon = 1e-8

    m = np.zeros_like(params)  # Inicializa o vetor de primeiro momento (m)
    v = np.zeros_like(params)  # Inicializa o vetor de segundo momento (v)

    for iteration in range(max_iter):
        aprox = Gerar_TSK(x, entrada, params, tipo)  # Gera as aproximações do modelo
        rmse = np.sqrt(np.mean((np.array(aprox) - np.array(fx))**2))  # Calcula o erro RMSE
        errors.append(rmse)

        if iteration > 0 and abs(errors[-1] - errors[-2]) < tol:  # Critério de convergência
            print(f"Converged after {iteration} iterations.")
            break

        gradients = np.zeros_like(params)  # Inicializa os gradientes

        for i, v_val in enumerate(x):  # Itera sobre os dados de entrada
            ws = entrada.calcular_pesos(v_val)  # Calcula os pesos fuzzy
            zs = [params[j][0] * v_val + params[j][1] if tipo == 1 else params[j][0] for j in range(len(params))]
            numerador = sum(ws[j] * zs[j] for j in range(len(params)))  # Soma ponderada
            denominador = sum(ws) + epsilon  # Evita divisão por zero
            y_hat = numerador / denominador  # Calcula a saída estimada
            error = y_hat - fx[i]  # Calcula o erro

            for j in range(len(params)):  # Calcula os gradientes
                dz = [v_val, 1] if tipo == 1 else [1]
                dnum = ws[j] * np.array(dz)
                dy_dparams = dnum / denominador
                gradients[j] += error * dy_dparams

        m = beta1 * m + (1 - beta1) * gradients  # Atualiza o primeiro momento
        v = beta2 * v + (1 - beta2) * (gradients ** 2)  # Atualiza o segundo momento
        m_hat = m / (1 - beta1 ** (iteration + 1))  # Correção do viés do primeiro momento
        v_hat = v / (1 - beta2 ** (iteration + 1))  # Correção do viés do segundo momento
        params -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)  # Atualiza os parâmetros

    return params, errors

# Função que implementa o gradiente descendente com RMSprop
def Gradient_Descent_RMSprop(x, fx, entrada, tipo, learning_rate=1e-3, max_iter=1000, tol=1e-5, beta=0.9):
    if tipo == 0:
        num_params = 1
    elif tipo == 1:
        num_params = 2
    else:
        raise ValueError("Tipo de Takagi-Sugeno não existe!")

    params = np.zeros((len(entrada.funcs), num_params))  # Inicializa os parâmetros como zeros
    errors = []
    epsilon = 1e-8

    v = np.zeros_like(params)  # Inicializa o acumulador RMSprop

    for iteration in range(max_iter):
        aprox = Gerar_TSK(x, entrada, params, tipo)  # Gera as aproximações do modelo
        rmse = np.sqrt(np.mean((np.array(aprox) - np.array(fx))**2))  # Calcula o erro RMSE
        errors.append(rmse)

        if iteration > 0 and abs(errors[-1] - errors[-2]) < tol:  # Critério de convergência
            print(f"Converged after {iteration} iterations.")
            break

        gradients = np.zeros_like(params)  # Inicializa os gradientes

        for i, v_val in enumerate(x):  # Itera sobre os dados de entrada
            ws = entrada.calcular_pesos(v_val)  # Calcula os pesos fuzzy
            zs = [params[j][0] * v_val + params[j][1] if tipo == 1 else params[j][0] for j in range(len(params))]
            numerador = sum(ws[j] * zs[j] for j in range(len(params)))  # Soma ponderada
            denominador = sum(ws) + epsilon  # Evita divisão por zero
            y_hat = numerador / denominador  # Calcula a saída estimada
            error = y_hat - fx[i]  # Calcula o erro

            for j in range(len(params)):  # Calcula os gradientes
                dz = [v_val, 1] if tipo == 1 else [1]
                dnum = ws[j] * np.array(dz)
                dy_dparams = dnum / denominador
                gradients[j] += error * dy_dparams

        v = beta * v + (1 - beta) * (gradients ** 2)  # Atualiza o acumulador de gradientes ao quadrado
        params -= learning_rate * gradients / (np.sqrt(v) + epsilon)  # Atualiza os parâmetros

    return params, errors
