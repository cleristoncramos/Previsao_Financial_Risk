"""
Textos "Como ler este gráfico" do painel.

Explicações fixas, em linguagem simples, para quem não é da área: o que o gráfico mostra,
como interpretá-lo e um exemplo com os números do ativo e do período selecionados. Os trechos
entre chaves ({ativo}, {pico}...) são preenchidos pelo painel no momento da exibição.

Para revisar a linguagem, basta editar os textos abaixo; os nomes entre chaves devem ser mantidos.
"""

TEXTOS_GRAFICOS = {
    # ------------------------------------------------------------------ Análise Exploratória
    "eda_precos": """
**O que mostra.** O preço de fechamento do ativo em cada pregão (linha azul) e duas médias móveis:
a média dos últimos 20 pregões (cerca de um mês) e a dos últimos 50 (cerca de dois meses e meio).

**Como ler.** A linha azul subindo indica valorização; descendo, desvalorização. As médias móveis
suavizam o sobe e desce diário e mostram a tendência: quando o preço fica acima delas, a tendência
recente é de alta; abaixo, de baixa. As linhas pontilhadas verticais marcam acontecimentos que
mexeram com o mercado; passe o mouse sobre elas para ver qual foi.

**Neste período.** O {ativo} foi de {preco_inicio} para {preco_fim}, uma variação de {variacao}.
""",
    "eda_volatilidade": """
**O que mostra.** A volatilidade, isto é, a intensidade com que o preço oscila. Ela é calculada com
os retornos dos últimos 21 pregões e expressa em termos anuais.

**Como ler.** Quanto mais alta a área, mais nervoso está o mercado. Uma volatilidade de 20% ao ano
corresponde a oscilações típicas de cerca de 1,3% por dia. Os picos costumam coincidir com crises e
notícias inesperadas: compare-os com as linhas pontilhadas dos eventos.

**Neste período.** O maior valor foi {pico} em {data_pico}. No último pregão do período, a
volatilidade estava em {atual}.
""",
    "eda_fatos": """
**O que mostra.** À esquerda, quantas vezes cada tamanho de retorno diário aconteceu (barras),
comparado com a curva em forma de sino da distribuição Normal (linha vermelha). À direita, o
Q-Q plot compara os retornos observados com os que a Normal preveria.

**Como ler.** Se os retornos fossem "normais", as barras acompanhariam a curva e os pontos do Q-Q
plot ficariam sobre a reta. Nas pontas, os pontos se afastam da reta: dias de queda ou alta muito
fortes acontecem com mais frequência do que a Normal prevê. É o que se chama de caudas pesadas, e
por isso medidas de risco baseadas na Normal tendem a subestimar perdas extremas.

**Neste período.** A maior queda em um dia foi de {pior_dia} em {data_pior}; a maior alta, de
{melhor_dia} em {data_melhor}.
""",
    "eda_correlacao": """
**O que mostra.** O quanto os retornos diários de cada par de ativos andam juntos, numa escala de
−1 a 1. Cada par aparece uma única vez.

**Como ler.** Perto de 1 (vermelho), os dois ativos costumam subir e cair nos mesmos dias. Perto de
0, um não ajuda a prever o outro. Valores negativos (azul) indicariam movimentos opostos. Ativos
pouco correlacionados ajudam a diversificar uma carteira.

**Neste período.** O par mais correlacionado foi {par_max} ({corr_max}); o menos correlacionado,
{par_min} ({corr_min}).
""",
    # ------------------------------------------------------------------ Desempenho dos Modelos
    "tabela_metricas": """
**O que mostra.** O erro de cada abordagem ao prever a volatilidade do pregão seguinte, medido no
período de teste: os 20% finais da série, que os modelos não viram no treino.

**Como ler.** RMSE e MAE são erros médios na escala da volatilidade: 0,01 equivale a 1 ponto
percentual. Quanto menores, melhor; o RMSE pesa mais os erros grandes. O R² vai até 1 e indica
quanto da variação real o modelo acompanha. O baseline de persistência é a regra mais simples
possível, "amanhã será igual a hoje", e serve de régua: um modelo só agrega valor se errar menos
do que ela. Um modelo com desempenho fraco também ensina algo: mostra que nem toda técnica
sofisticada funciona melhor.

**Neste ativo.** O menor RMSE foi o de {melhor_modelo} ({melhor_rmse}).
""",
    "real_vs_modelos": """
**O que mostra.** A volatilidade que de fato ocorreu (linha preta) e as previsões dos modelos
escolhidos na lista acima, ao longo do período de teste.

**Como ler.** Quanto mais colada à linha preta, melhor a previsão. Repare que as previsões
costumam reagir um pouco depois dos saltos: elas se baseiam no que já aconteceu. Para comparar
com calma, deixe poucos modelos selecionados e use o zoom do gráfico (arraste sobre uma região).
""",
    "validacao_cruzada": """
**O que mostra.** O erro do XGBoost em cinco testes sucessivos: em cada um, o modelo treina com
todo o passado disponível e é avaliado no bloco de tempo seguinte.

**Como ler.** Barras parecidas indicam um modelo estável ao longo do tempo. Uma barra muito mais
alta revela um período difícil de prever, em geral uma crise que o modelo nunca tinha visto no
treino, como a pandemia de 2020.
""",
    "curva_mlp": """
**O que mostra.** Como o erro da Rede Neural diminuiu a cada rodada de treino (época), nos dados
de treino e numa parte separada para validação.

**Como ler.** As duas curvas devem cair. Se a de validação passa a subir enquanto a de treino
continua caindo, a rede começou a "decorar" os dados em vez de aprender o padrão. A linha
tracejada marca a época com menor erro de validação, cujos pesos foram mantidos.
""",
    # ------------------------------------------------------------------ Interpretabilidade
    "shap_bar": """
**O que mostra.** Quais informações mais pesam, em média, na previsão do XGBoost.

**Como ler.** Cada barra é uma variável de entrada; quanto maior, mais ela altera a volatilidade
prevista. Se a volatilidade atual domina, o modelo está, na prática, apoiado na ideia de que a
volatilidade de amanhã se parece com a de hoje.

**Neste ativo.** A variável mais importante é {top_feature}.
""",
    "shap_beeswarm": """
**O que mostra.** O efeito de cada variável em cada pregão do teste: cada ponto é um pregão.

**Como ler.** Pontos à direita da linha central aumentaram a volatilidade prevista; à esquerda,
reduziram. A cor indica o valor da variável naquele dia: vermelho para valores altos, azul para
baixos. Se os pontos vermelhos ficam à direita, valores altos daquela variável levam a previsões
de mais risco.
""",
    "previsao_local": """
**O que mostra.** A previsão do XGBoost para um pregão escolhido por você, comparada com a
volatilidade que realmente ocorreu no dia seguinte.

**Como ler.** Arraste o controle para escolher a data. O erro é a diferença entre o previsto e o
real: positivo quando o modelo superestimou o risco, negativo quando subestimou.
""",
    "explicacoes_locais": """
**O que mostra.** Por que o modelo chegou àquela previsão no pregão escolhido.

**Como ler.** No gráfico em cascata (SHAP), a conta começa no valor médio previsto pelo modelo e
cada barra empurra a previsão para cima (vermelho) ou para baixo (azul), até chegar ao valor final.
O LIME (no modo pesquisador) mostra o mesmo raciocínio de outra forma: quais condições do dia, como
"volatilidade acima de tal nível", mais influenciaram a previsão.
""",
    # ------------------------------------------------------------------ Simulador de Risco
    "projecao": """
**O que mostra.** A volatilidade prevista para o próximo pregão, com base nos dados mais recentes,
e em que faixa de risco ela cai.

**Como ler.** A faixa é comparada com o histórico do próprio ativo: "baixo" fica entre os 25% dias
mais calmos da série, "moderado" na metade central, "elevado" entre os 25% mais agitados e
"extremo" entre os 5% mais turbulentos. Assim, uma ação naturalmente mais volátil não aparece
sempre como "extrema" só por ser comparada a um índice.
""",
    "var": """
**O que mostra.** O Value at Risk (VaR): uma estimativa da perda que não deve ser ultrapassada em
um dia, para o valor investido que você informar.

**Como ler.** Um VaR de 95% igual a R$ 1.000 significa que, em 95 de cada 100 pregões, a perda
esperada não passa de R$ 1.000. Não é a perda máxima possível: nos outros 5 dias, ela pode ser
maior. O cálculo supõe retornos com distribuição Normal, que subestima eventos extremos. É uma
ferramenta de estudo, não uma recomendação de investimento.
""",
}


def texto_grafico(chave: str, **valores) -> str:
    """Texto da chave, com os campos entre chaves preenchidos (os ausentes ficam como "—")."""

    class _Padrao(dict):
        def __missing__(self, campo):
            return "—"

    return TEXTOS_GRAFICOS[chave].strip().format_map(_Padrao(valores))
