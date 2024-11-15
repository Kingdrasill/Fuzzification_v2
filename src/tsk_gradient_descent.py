from src.domain import *
from scipy.optimize import minimize

def Gerar_TSK(x, entrada, params, tipo):
    y = []
    for v in x:
        ws = entrada.calcular_pesos(v)
        if tipo == 0:
            zs = [(params[i][0]) for i in range(len(params))]
        elif tipo == 1:
            zs = [(params[i][0] * v + params[i][1]) for i in range(len(params))]
        else:
            return "Tipo de Takagi-Sugeno não existe!"
        numerador = sum([ws[i] * zs[i] for i in range(len(params))])
        denominador = sum(ws)
        y.append(numerador / denominador)
    return y

def RMSE_Objective(flat_params, x, fx, entrada, num_params, tipo, errors):
    params = flat_params.reshape(-1, num_params)
    aprox = Gerar_TSK(x, entrada, params, tipo)
    rmse = (np.sqrt(np.square(np.array(aprox) - np.array(fx)))).mean()
    errors.append(rmse)
    return rmse

def Gerar_Resultado_Aproximado(x, fx, entrada, method, tipo):
    if tipo == 0:
        num_params = 1
    elif tipo == 1:
        num_params = 2
    else:
        return ("Tipo de Takagi-Sugeno passado não existe!", None)
    num_params = 2
    params = [np.zeros(num_params) for i in range(len(entrada.funcs))]
    erros = []

    initial_params = np.array([param for sublist in params for param in sublist]).flatten()

    result = minimize(
        RMSE_Objective,
        initial_params,
        args=(x, fx, entrada, num_params, tipo, erros),
        method=method,
        tol=1e-5,
        options={'maxiter': 1000, 'disp': True}
    )
    optimized_params = result.x.reshape(-1, num_params)

    aprox_optimized = Gerar_TSK(x, entrada, optimized_params, 1)

    return (aprox_optimized, erros)