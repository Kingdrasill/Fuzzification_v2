from src.tsk_gradient_descent import *
from matplotlib.gridspec import GridSpec

inf, sup = 0, 10
x = np.linspace(inf, sup, 200)
fx = []
for value in x:
    fx.append(math.exp(- value / 5) * math.sin(3 * value) + 0.5 * math.sin(value))

def Gerar_Graficos(x, fx, entrada, aproxs, names, erros, filename):
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
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
    
    ax2.plot(x, fx, label='Real')
    for aprox, name in zip(aproxs, names):
        ax2.plot(x, aprox, label=name)
    ax2.set_title('Resposta Real X Aproximação')
    ax2.legend()
    ax2.grid(True)

    for erro, name in zip(erros, names):
        ax3.plot(erro, label=name)
    for erro, name in zip(erros, names):
        ax3.scatter(len(erro) - 1, erro[-1], marker='x')
    ax3.set_title('Evolução do RMSE')
    ax3.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

for t in ['TR', 'TP', 'GS', 'SN', 'CC', 'LP']:
    for qtd in range(4, 8):
        aproxs = []
        names = []
        erros = []
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
        
        for m in ['CG', 'BFGS']:
            for o in [0, 1]:
                aprox, erro = Gerar_Resultado_Aproximado(x, fx, entrada, m, o)
                aproxs.append(aprox)
                names.append(f'Aprox {m} {o} ordem')
                erros.append(erro)
        Gerar_Graficos(x, fx, entrada, aproxs, names, erros, 'data/' + filename + '.png')