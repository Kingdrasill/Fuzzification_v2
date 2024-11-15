from src.functions import *
import random as rd
import matplotlib.pyplot as plt
import numpy as np

# Classe usada para guardar informações de um domínio
class Domain:
    # A classe possui limites inf e sup, um nome e as informações das funções
    def __init__(self, inferior, superior, funcs, name):
        self.inf = inferior
        self.sup = superior
        self.funcs = funcs
        self.name = name

    # Calcula e retorna os graus de ativação de uma amostra, a chamada da função depende de seu tipo
    def calcular_pesos(self, x):
        graus = []
        for f in self.funcs:
            match f['tipo']:
                case w if w in ['GS', 'SG', 'SS', 'ZS', 'CC', 'RT', 'LP']:
                    graus.append(f['func'](f['values'][0], f['values'][1], x))
                case w if w in ['TR', 'SN', 'GD']:
                    graus.append(f['func'](f['values'][0], f['values'][1], f['values'][2], x))
                case w if w in ['TP']:
                    graus.append(f['func'](f['values'][0], f['values'][1], f['values'][2], f['values'][3], x))
        return graus

# Gera a entrada do sistema fuzzy com qtd funções triangulares
def Gerar_Entrada_TR(inf, sup, qtd):
    # Gera os centros máximos das funções
    centers = np.linspace(inf, sup, qtd)
    # Armazena todos as funções
    funcs = []

    max = 3
    min = 1.3
    i = 0
    while i < qtd:
        b = centers[i]
        i += 1
        # Gera aletoriamente os valores que não são o centro
        a = b - rd.uniform(min, max)
        c = b + rd.uniform(min, max)
        # Cada função tem seu nome, tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{i}',
            'tipo': 'TR',
            'func': Calcular_Mi_TR,
            'values': [a, b, c]
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain

# Gera a entrada do sistema fuzzy com qtd funções trapezoidais
def Gerar_Entrada_TP(inf, sup, qtd):
    # Gera os pontos máximos das funções
    points = np.linspace(inf, sup, qtd*2)
    # Armazena todos as funções
    funcs = []

    max = 2.5
    min = 1
    i = 0
    while i < qtd * 2:
        b = points[i]
        c = points[i+1]
        i += 2
        # Gera aletoriamente os valores que não são do intervalo central
        a = b - rd.uniform(min, max)
        d = c + rd.uniform(min, max)
        # Cada função tem seu nome, tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{int(i/2)}',
            'tipo': 'TP',
            'func': Calcular_Mi_TP,
            'values': [a, b, c, d]
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain

# Gera a entrada do sistema fuzzy com qtd funções gaussianas
def Gerar_Entrada_GS(inf, sup, qtd):
    # Gera os centros máximos das funções
    centers = np.linspace(inf, sup, qtd)
    # Armazena todos as funções
    funcs = []

    max = 2
    min = 0.5
    i = 0
    while i < qtd:
        c = centers[i]
        i += 1
        # Gera aletoriamente o valor que não é o centro
        sigma = rd.uniform(min, max)
        # Cada função tem seu nome, tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{i}',
            'tipo': 'GS',
            'func': Calcular_Mi_GS,
            'values': [c, sigma]
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain


# Gera a entrada do sistema fuzzy com qtd funções sinos generalizadas
def Gerar_Entrada_SN(inf, sup, qtd):
    # Gera os pontos máximos das funções
    points = np.linspace(inf, sup, qtd*2)
    # Armazena todos as funções
    funcs = []

    max = 3
    min = 0.5
    # Gera aletoriamente o valor que não é o centro e é igual para todas funções
    b = rd.uniform(min, max)
    i = 0
    while i < qtd * 2:
        p1 = points[i]
        p2 = points[i+1]
        i += 2
        c = (p1 + p2) / 2
        a = abs(p2 - p1)
        # Cada função tem seu nome, tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{int(i/2)}',
            'tipo': 'SN',
            'func': Calcular_Mi_SN,
            'values': [a, b, c]
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain

# Gera a entrada do sistema fuzzy com qtd funções cauchys
def Gerar_Entrada_CC(inf, sup, qtd):
    # Gera os centros máximos das funções
    centers = np.linspace(inf, sup, qtd)
    # Armazena todos as funções
    funcs = []
    
    max = 0.85
    min = 0.25
    # Gera aletoriamente o valor que não é o centro que é igual para todas as funções
    gamma = rd.uniform(min, max)
    i = 0
    while i < qtd:
        x_0 = centers[i]
        i += 1
        # Cada função tem seu tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{i}',
            'tipo': 'CC',
            'func': Calcular_Mi_CC,
            'values': [x_0, gamma]
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain

# Gera a entrada do sistema fuzzy com qtd funções laplaces
def Gerar_Entrada_LP(inf, sup, qtd):
    # Gera os centros máximos das funções
    centers = np.linspace(inf, sup, qtd)
    # Armazena todos as funções
    funcs = []
    
    max = 1
    min = 0.25
    # Gera aletoriamente o valor que não é o centro que é igual para todas as funções
    b = rd.uniform(min, max)
    i = 0
    while i < qtd:
        mi = centers[i]
        i += 1
        # Cada função tem seu tipo, valores e qual a função a ser chamada
        funcs.append({
            'nome': f'f{i}',
            'tipo': 'LP',
            'func': Calcular_Mi_LP,
            'values': [mi, b],
        })
    domain = Domain(inf, sup, funcs, f'Funções da Entrada X')
    return domain