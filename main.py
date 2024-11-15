from src.tsk_gradient_descent import *
from matplotlib.gridspec import GridSpec

# Método para gerar os gráicos
def Gerar_Graficos(x, fx, entrada, aproxs, names, erros, filename):
    # Gera 3 gráficos na mesma linha
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])

    # Primeiro gráfico: Entrada do sistema fuzzy
    for f in entrada.funcs:
        y = []
        match f['tipo']:
            case w if w in ['GS', 'SG', 'SS', 'ZS', 'CC', 'RT', 'LP']:
                for v in x:
                    y.append(f['func'](f['values'][0], f['values'][1], v))
            case w if w in ['TR', 'SN', 'GD']:
                for v in x:
                    y.append(f['func'](f['values'][0], f['values'][1], f['values'][2], v))
            case w if w in ['TP']:
                for v in x:
                    y.append(f['func'](f['values'][0], f['values'][1], f['values'][2], f['values'][3], v))
        ax1.plot(x, y, label=f['nome'])
    ax1.set_title(entrada.name)
    ax1.legend()
    ax1.grid(True)
    
    colors = ['red', 'orange', 'green', 'purple']

    # Segundo gráfico: Resultado Real X Aproximado
    ax2.plot(x, fx, label='Real')
    for aprox, name, color in zip(aproxs, names, colors):
        ax2.plot(x, aprox, label=name, color=color)
    ax2.set_title('Resposta Real X Aproximação')
    ax2.legend()
    ax2.grid(True)

    # Terceiro gráfico: Evolução do RMSE
    for erro, name, color in zip(erros, names, colors):
        ax3.plot(erro, label=name, color=color)
    for erro, name, color in zip(erros, names, colors):
        ax3.scatter(len(erro) - 1, erro[-1], color=color, marker='x')
    ax3.set_title('Evolução do RMSE')
    ax3.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

max_iter = 5000
# Limites do sistema fuzzy
inf, sup = 0, 10
# Intervalo de valores
x = np.linspace(inf, sup, 300)
# Gera o resultado real
fx = []
for value in x:
    fx.append(math.exp(- value / 5) * math.sin(3 * value) + 0.5 * math.sin(value))

# Gera entradas de funções do tipo: triangular, trapezoidal, gaussiana, sino, cauchy, laplace
for t in ['TR', 'TP', 'GS', 'SN', 'CC', 'LP']:
    # Quantidade de funções de pertinência: 4 a 8
    for qtd in range(4, 9):
        match t:
            case 'TR':
                entrada = Gerar_Entrada_TR(inf, sup, qtd)
                filename = 'domain_tr_{}'.format(qtd)
            case 'TP':
                entrada = Gerar_Entrada_TP(inf, sup, qtd)
                filename = 'domain_tp_{}'.format(qtd)
            case 'GS':
                entrada = Gerar_Entrada_GS(inf, sup, qtd)
                filename = 'domain_gs_{}'.format(qtd)
            case 'SN':
                entrada = Gerar_Entrada_SN(inf, sup, qtd)
                filename = 'domain_sn_{}'.format(qtd)
            case 'CC':
                entrada = Gerar_Entrada_CC(inf, sup, qtd)
                filename = 'domain_cc_{}'.format(qtd)
            case 'LP':
                entrada = Gerar_Entrada_LP(inf, sup, qtd)
                filename = 'domain_lp_{}'.format(qtd)
        
        for ordem in [0, 1]:
            aproxs = []
            names = []
            erros = []

            params, erro = Gradient_Descent_Momentum(x, fx, entrada, ordem, max_iter=max_iter)
            aprox = Gerar_TSK(x, entrada, params, ordem)
            aproxs.append(aprox)
            names.append(f'Aprox GD Momentum')
            erros.append(erro)

            params, erro = Gradient_Descent_Adam(x, fx, entrada, ordem, max_iter=max_iter)
            aprox = Gerar_TSK(x, entrada, params, ordem)
            aproxs.append(aprox)
            names.append(f'Aprox GD Adam')
            erros.append(erro)

            params, erro = Gradient_Descent_RMSprop(x, fx, entrada, ordem, max_iter=max_iter)
            aprox = Gerar_TSK(x, entrada, params, ordem)
            aproxs.append(aprox)
            names.append(f'Aprox GD RMSprop')
            erros.append(erro)

            aprox, erro = Gerar_Resultado_Aproximado(x, fx, entrada, 'BFGS', ordem, max_iter=max_iter)
            aproxs.append(aprox)
            names.append(f'Aprox Minimize BFGS')
            erros.append(erro)

            Gerar_Graficos(x, fx, entrada, aproxs, names, erros, f'imgs/{ordem}-ordem/' + filename + '.png')

print("Os gráficos estão na pasta imgs")