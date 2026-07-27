"""
streamlit_app.py — Interface visual premium de previsão de churn.
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Permitir importar o módulo partilhado em src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from predict import (  # noqa: E402
    CATEGORICAL_OPTIONS,
    EXEMPLO_CLIENTE,
    FEATURE_ORDER,
    NOMES_PARAMETROS,
    analisar_parametros,
    prever,
)

st.set_page_config(page_title="Telco Churn Analytics", page_icon="📶", layout="wide")

# CSS Personalizado para aprimorar o tema
st.markdown("""
<style>
    /* Estilizar barra de progresso */
    .stProgress .st-bo { background-color: #F94C10; }
    /* Cor de realce das métricas */
    div[data-testid="stMetricValue"] { color: #F94C10; }
</style>
""", unsafe_allow_html=True)

# -----------------
# Sidebar Navigation
# -----------------
with st.sidebar:
    st.title("Telco Churn Analytics")
    st.markdown("Bem-vindo ao painel corporativo de previsão de rotatividade (Churn) de clientes.")
    st.divider()
    modo = st.radio(
        "Selecione o Módulo de Visualização:",
        [
            "Previsor de Churn (Individual)",
            "Previsão Completa (19 Parâmetros)",
            "Visão Global (Analítica)",
        ]
    )
    st.divider()
    st.caption("Desenvolvido por Hélder Traquinho")
    st.caption("[GitHub Repository](https://github.com/heltrakinho07/telco-churn-prediction)")

def selecionar(label, chave, prefixo=""):
    """Cria um menu para uma variável categórica, com valor-exemplo por defeito."""
    opcoes = CATEGORICAL_OPTIONS[chave]
    idx = opcoes.index(EXEMPLO_CLIENTE[chave]) if EXEMPLO_CLIENTE[chave] in opcoes else 0
    return st.selectbox(label, opcoes, index=idx, key=f"{prefixo}{chave}")


# Os 19 parâmetros do modelo, agrupados para o formulário completo
GRUPOS_PARAMETROS = {
    "Perfil demográfico": ["gender", "SeniorCitizen", "Partner", "Dependents"],
    "Serviços contratados": [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies",
    ],
    "Contrato, faturação e valores": [
        "Contract", "PaperlessBilling", "PaymentMethod", "tenure",
        "MonthlyCharges", "TotalCharges",
    ],
}
assert sorted(sum(GRUPOS_PARAMETROS.values(), [])) == sorted(FEATURE_ORDER)

LIMITES_NUMERICOS = {
    "tenure": (0, 72, 1),
    "MonthlyCharges": (0.0, 500.0, 5.0),
    "TotalCharges": (0.0, 15000.0, 100.0),
}


def campo_parametro(chave):
    """Cria o widget certo para um parâmetro do modelo (categórico ou numérico)."""
    label = NOMES_PARAMETROS[chave]
    if chave in CATEGORICAL_OPTIONS:
        return selecionar(label, chave, prefixo="full_")
    minimo, maximo, passo = LIMITES_NUMERICOS[chave]
    return st.number_input(
        label, minimo, maximo, type(minimo)(EXEMPLO_CLIENTE[chave]),
        step=passo, key=f"full_{chave}",
    )


@st.cache_data(ttl="15m", max_entries=50, show_spinner="A avaliar os 19 parâmetros...")
def analisar_cliente(dados: dict) -> dict:
    """Previsão + peso de cada parâmetro (em cache: o mesmo perfil não recalcula)."""
    return analisar_parametros(dados)

def render_gauge(prob, risk_level):
    """Renderiza um gráfico de velocímetro de alta qualidade usando Plotly."""
    color = "green" if risk_level == "Baixo" else ("#FFC300" if risk_level == "Médio" else "red")
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        number = {"suffix": "%", "font": {"color": "#E0E0E0"}},
        title = {'text': f"Risco de Churn: {risk_level}", 'font': {'size': 20, "color": "#E0E0E0"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.1)'},
                {'range': [30, 70], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.1)'}],
        }
    ))
    fig.update_layout(paper_bgcolor="#1E1E1E", font={'color': "#E0E0E0"}, height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig


if modo == "Previsor de Churn (Individual)":
    st.title("Motor de Previsão de Churn")
    st.markdown("Insira os dados do cliente abaixo para gerar a análise de risco baseada em Machine Learning (HistGradientBoosting - ROC-AUC 0.84).")
    
    with st.form("cliente_form"):
        # Organização em Abas (Tabs) melhora imensamente a interface
        tab1, tab2, tab3 = st.tabs(["👤 Perfil Pessoal", "⚙️ Serviços de Comunicação", "💳 Contrato & Faturação"])
        
        with tab1:
            st.markdown("#### Detalhes Demográficos")
            c1, c2 = st.columns(2)
            with c1:
                gender = selecionar("Género", "gender")
                SeniorCitizen = selecionar("É Idoso (65+)?", "SeniorCitizen")
            with c2:
                Partner = selecionar("Possui Parceiro/a?", "Partner")
                Dependents = selecionar("Possui Dependentes?", "Dependents")
                
        with tab2:
            st.markdown("#### Pacotes e Serviços")
            c3, c4, c5 = st.columns(3)
            with c3:
                PhoneService = selecionar("Serviço Telefónico", "PhoneService")
                MultipleLines = selecionar("Múltiplas Linhas", "MultipleLines")
                InternetService = selecionar("Serviço de Internet", "InternetService")
            with c4:
                OnlineSecurity = selecionar("Segurança Online", "OnlineSecurity")
                OnlineBackup = selecionar("Backup Online", "OnlineBackup")
                DeviceProtection = selecionar("Proteção de Equipamento", "DeviceProtection")
            with c5:
                TechSupport = selecionar("Suporte Técnico Premium", "TechSupport")
                StreamingTV = selecionar("Subscrição Streaming TV", "StreamingTV")
                StreamingMovies = selecionar("Subscrição Filmes", "StreamingMovies")
                
        with tab3:
            st.markdown("#### Retenção e Valores")
            c6, c7 = st.columns(2)
            with c6:
                Contract = selecionar("Tipo de Contrato", "Contract")
                PaymentMethod = selecionar("Método de Pagamento", "PaymentMethod")
                PaperlessBilling = selecionar("Fatura Eletrónica (Paperless)", "PaperlessBilling")
            with c7:
                tenure = st.slider("Tempo de Casa (meses)", 0, 72, EXEMPLO_CLIENTE["tenure"])
                MonthlyCharges = st.number_input("Encargo Mensal (MZN)", 0.0, 500.0, float(EXEMPLO_CLIENTE["MonthlyCharges"]))
                TotalCharges = st.number_input("Total Gasto (MZN)", 0.0, 15000.0, float(EXEMPLO_CLIENTE["TotalCharges"]))

        st.markdown("<br>", unsafe_allow_html=True)
        submeter = st.form_submit_button("Analisar Risco de Churn", type="primary", width="stretch")

    if submeter:
        dados = {
            "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
            "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
            "MultipleLines": MultipleLines, "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
            "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
            "Contract": Contract, "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }
        r = prever(dados)
        prob = r["probabilidade"]
        risco = r["risco"]

        st.markdown("---")
        st.subheader("Resultado da Avaliação")
        
        col_res1, col_res2 = st.columns([1, 1])
        with col_res1:
            st.plotly_chart(render_gauge(prob, risco), width="stretch")
            
        with col_res2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if r["churn"]:
                st.error("🚨 ALERTA: Cliente com elevada probabilidade de abandono. Sugere-se alocar para a equipa de retenção e oferecer um pacote de fidelização imediato.")
                st.metric(label="Recomendação do Sistema", value="Ação Imediata")
            else:
                st.success("✅ SEGURO: Este cliente apresenta forte lealdade ao serviço. O risco de cancelamento atual é mínimo.")
                st.metric(label="Recomendação do Sistema", value="Manter Estratégia")
                
            st.info(f"O modelo de Machine Learning atribui uma certeza estatística de **{prob*100:.1f}%** na ocorrência de Churn perante o perfil analisado.")

elif modo == "Previsão Completa (19 Parâmetros)":
    st.title("Previsão Completa (Todos os Parâmetros)")
    st.markdown(
        "Este módulo usa **os 19 parâmetros do modelo em simultâneo**: prevê o risco do cliente e, "
        "mantendo o resto do perfil fixo, volta a prever variando cada parâmetro para medir quanto "
        "cada um pesa neste caso concreto."
    )

    with st.form("form_completo"):
        dados = {}
        for grupo, campos in GRUPOS_PARAMETROS.items():
            with st.container(border=True):
                st.markdown(f"#### {grupo}")
                colunas = st.columns(4)
                for i, campo in enumerate(campos):
                    with colunas[i % 4]:
                        dados[campo] = campo_parametro(campo)

        submeter_completo = st.form_submit_button(
            "Executar previsão com todos os parâmetros", type="primary", width="stretch"
        )

    if submeter_completo:
        analise = analisar_cliente(dados)
        prob = analise["probabilidade"]

        st.markdown("---")
        st.subheader("Resultado com o perfil completo")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Probabilidade de Churn", f"{prob*100:.1f}%")
        m2.metric("Nível de Risco", analise["risco"])
        m3.metric("Decisão do Modelo", "Vai sair" if analise["churn"] else "Fica")
        m4.metric("Parâmetros Considerados", f"{len(analise['parametros'])}/19")

        col_g, col_b = st.columns([1, 1])
        with col_g:
            st.plotly_chart(render_gauge(prob, analise["risco"]), width="stretch")

        tabela = pd.DataFrame(analise["parametros"])
        tabela["Influência (pp)"] = (tabela["influencia"] * 100).round(1)
        tabela["Ganho Possível (pp)"] = (tabela["ganho"] * 100).round(1)
        # Estas colunas misturam texto e números (ex.: "Two year" e 72 meses)
        tabela[["valor_atual", "melhor_valor"]] = tabela[["valor_atual", "melhor_valor"]].astype(str)

        with col_b:
            fig_infl = px.bar(
                tabela.head(10).sort_values("Influência (pp)"),
                x="Influência (pp)", y="nome", orientation="h",
                title="Peso de cada parâmetro nesta previsão (top 10)",
                color="Influência (pp)", color_continuous_scale="Oranges",
                labels={"nome": ""},
            )
            fig_infl.update_layout(
                paper_bgcolor="#1E1E1E", plot_bgcolor="#1E1E1E",
                font={"color": "#E0E0E0"}, height=350, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_infl, width="stretch")

        st.markdown("#### Detalhe de todos os parâmetros usados")
        st.caption(
            "Influência = diferença entre o pior e o melhor cenário desse parâmetro. "
            "Ganho possível = redução do risco atual se o parâmetro passasse ao melhor valor."
        )
        st.dataframe(
            tabela[["nome", "valor_atual", "Influência (pp)", "melhor_valor", "Ganho Possível (pp)"]],
            hide_index=True,
            width="stretch",
            column_config={
                "nome": st.column_config.TextColumn("Parâmetro"),
                "valor_atual": st.column_config.TextColumn("Valor do cliente"),
                "Influência (pp)": st.column_config.ProgressColumn(
                    "Influência (pp)", format="%.1f", min_value=0.0,
                    max_value=float(max(tabela["Influência (pp)"].max(), 1.0)),
                ),
                "melhor_valor": st.column_config.TextColumn("Melhor valor possível"),
                "Ganho Possível (pp)": st.column_config.NumberColumn(
                    "Ganho possível (pp)", format="%.1f"
                ),
            },
        )

        # Plano de retenção: aplicar em conjunto as alavancas com ganho real
        alavancas = [p for p in analise["parametros"] if p["ganho"] > 0.01][:3]
        st.markdown("#### Plano de retenção sugerido")
        if not alavancas:
            st.success(
                "Nenhum parâmetro isolado reduz de forma relevante o risco deste cliente — "
                "o perfil já está no melhor cenário possível para o modelo."
            )
        else:
            cenario = dict(dados)
            for p in alavancas:
                cenario[p["parametro"]] = p["melhor_valor"]
            prob_simulada = prever(cenario)["probabilidade"]

            for p in alavancas:
                st.write(
                    f"- **{p['nome']}**: passar de `{p['valor_atual']}` para "
                    f"`{p['melhor_valor']}` → −{p['ganho']*100:.1f} pontos percentuais"
                )
            s1, s2, s3 = st.columns(3)
            s1.metric("Risco Atual", f"{prob*100:.1f}%")
            s2.metric(
                "Risco Após o Plano", f"{prob_simulada*100:.1f}%",
                delta=f"{(prob_simulada - prob)*100:.1f} pp", delta_color="inverse",
            )
            s3.metric("Redução Combinada", f"{(prob - prob_simulada)*100:.1f} pp")
            st.info(
                "Simulação obtida aplicando em simultâneo as alavancas acima e "
                "voltando a correr o modelo com os 19 parâmetros.", icon="🧪"
            )

elif modo == "Visão Global (Analítica)":
    st.title("Visão Global de Churn (Exemplo Analítico)")
    st.markdown("Este painel demonstra como os dados globais de Churn podem ser agregados para tomada de decisão estratégica executiva.")
    
    st.markdown("---")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Total de Clientes Ativos", "7,043", "+124 este mês")
    c_m2.metric("Taxa Global de Churn", "26.5%", "-1.2% (vs last year)", delta_color="inverse")
    c_m3.metric("MRR (Revenue MZN)", "456,230", "+5.4%")
    c_m4.metric("Custo Médio de Retenção", "12,450", "-300")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("O que faz os clientes saírem? (Drivers de Churn)")
    
    # Mock data for feature importance demonstration
    features = pd.DataFrame({
        "Fator": ["Contrato Mensal", "Internet Fibra Ótica", "Tempo de Casa Baixo", "Pagamento Check Eletrónico", "Ausência de Suporte Técnico"],
        "Impacto Relativo (%)": [35, 25, 20, 12, 8]
    })
    
    fig2 = px.bar(features, x="Impacto Relativo (%)", y="Fator", orientation='h', 
                  title="Feature Importance (Interpretação do Modelo HistGradientBoosting)", 
                  color="Impacto Relativo (%)", color_continuous_scale="Oranges")
    fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212", font={'color': "#E0E0E0"})
    st.plotly_chart(fig2, width="stretch")
    
    st.info("💡 **Insight de Negócio:** Clientes com contratos 'Month-to-Month' (Mensais) têm a probabilidade mais alta de Churn. Considere campanhas de desconto para incentivar transições para contratos Anuais.", icon="📈")
