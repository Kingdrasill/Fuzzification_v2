import math

# Calcula o grau de pertinência do x passado para uma função triangular de limites
# inf a sup e variáveis de forma a, b e c
def Calcular_Mi_TR(a, b, c, x):
    if a > b or c < b:
        return "Algum valor de variáveis de forma está errado"
    
    if (x <= a):
        return 0
    elif (x > a and x <= b):
        return ((x - a) / (b - a))
    elif (x > b and x <= c):
        return ((c - x) / (c - b))
    elif (x > c):
        return 0

# Calcula o grau de pertinência do x passado para uma função trapezoidal de limites
# inf a sup e variáveis de forma a, b, c e d
def Calcular_Mi_TP(a, b, c, d, x):
    if a > b or b > c or c > d:
        return "Algum valor de variáveis de forma está errado"
    
    if (x <= a):
        return 0
    elif (x > a and x <= b):
        return ((x - a) / (b - a))
    elif (x > b and x <= c):
        return 1
    elif (x > c and x <= d):
        return ((d - x) / (d - c))
    elif (x > d):
        return 0

# Calcula o grau de pertinência do x passado para uma função gaussiana de limites
# inf a sup e variáveis de forma c e sigma
def Calcular_Mi_GS(c, sigma, x):
    if (sigma == 0):
        return "Algum valor de variáveis de forma está errado"

    return math.exp(-((x - c) ** 2)/(2 * sigma ** 2))

# Calcula o grau de pertinência do x passado para uma função sino de limites
# inf a sup e variáveis de forma a, b e c
def Calcular_Mi_SN(a, b, c, x):
    if (a == 0):
        return "Algum valor de variáveis de forma está errado"

    return (1 / (1 + (abs((x - c) / a)**(2 * b))))

# Calcula o grau de pertinência do x passado para uma função cauchy de limites
# inf a sup e variáveis de forma x_0 e gamma
def Calcular_Mi_CC(x_0, gamma, x):
    if (gamma == 0):
        return "Algum valor de variáveis de forma está errado"

    return (1 / ((math.pi * gamma) * (1 + ((x - x_0) / gamma) ** 2)))

# Calcula o grau de pertinência do x passado para uma função laplace de limites
# inf a sup e variáveis de forma mi e b
def Calcular_Mi_LP(mi, b, x):
    if (b == 0):
        return "Algum valor de variáveis de forma está errado"
    
    return ((1 / (2 * b)) * math.exp(- (abs(x - mi)) / b))