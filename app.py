"""
Aplicativo de previsão de ações.

Execute com:
    streamlit run app.py
"""

import base64
import json
import os
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analista_ia as ia
import contexto as cx
import historico as hist
import motor_previsao as mp
import pagamento as pg

AZUL, LARANJA, VERDE_AGUA, VIOLETA = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
HORIZONTES = {1: "1 dia", 5: "1 semana", 10: "2 semanas", 21: "1 mês", 63: "3 meses"}
ARQ_SEGREDOS = Path(__file__).with_name(".streamlit") / "secrets.toml"

st.set_page_config(page_title="Previsão de Ações", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner="Baixando cotações...")
def carregar_yahoo(ticker: str, anos: int):
    return mp.baixar_yahoo(ticker, anos)


@st.cache_data(ttl=3600, show_spinner="Baixando dados externos (índices, dólar, juros, Focus, incerteza política)...")
def carregar_externos(ticker: str, inicio, fim):
    return mp.baixar_externos(ticker, inicio, fim)


@st.cache_data(ttl=1800, show_spinner="Buscando notícias, consenso de analistas e boletim Focus...")
def carregar_contexto(ticker: str):
    return cx.coletar(ticker)


def fmt_preco(v: float) -> str:
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(v: float, sinal: bool = True) -> str:
    return (f"{v:+.2f}%" if sinal else f"{v:.2f}%").replace(".", ",")


def desc_h(h: int) -> str:
    return HORIZONTES.get(h, f"{h} pregões")


def segredo(nome: str, padrao=""):
    """Lê uma configuração: variável de ambiente, segredos do Streamlit (nuvem) ou arquivo local."""
    if os.environ.get(nome):
        return os.environ[nome]
    try:
        if nome in st.secrets:
            return st.secrets[nome]
    except Exception:
        pass
    try:
        return tomllib.loads(ARQ_SEGREDOS.read_text(encoding="utf-8")).get(nome, padrao)
    except Exception:
        return padrao


def salvar_chave(chave: str):
    ARQ_SEGREDOS.parent.mkdir(exist_ok=True)
    ARQ_SEGREDOS.write_text(f'ANTHROPIC_API_KEY = "{chave}"\n', encoding="utf-8")


# Modo público: app publicado na internet (Streamlit Community Cloud roda os apps em /mount/src).
# Nele, a chave do dono nunca vai para o navegador, visitantes não gravam nada no servidor
# e o uso da chave do dono exige senha e tem limite diário.
MODO_PUBLICO = str(segredo("MODO_PUBLICO", "")).lower() in ("1", "true", "sim") \
    or Path(__file__).resolve().as_posix().startswith("/mount/src/")
CHAVE_DONO = segredo("ANTHROPIC_API_KEY")
SENHA_APP = str(segredo("SENHA_APP", ""))
LIMITE_DIARIO = int(segredo("LIMITE_ANALISES_DIA", 10))
ARQ_USO = Path(__file__).with_name("uso_ia.json")


def usos_hoje() -> int:
    try:
        uso = json.loads(ARQ_USO.read_text())
        return uso["n"] if uso["dia"] == str(pd.Timestamp.today().date()) else 0
    except Exception:
        return 0


def registrar_uso():
    try:
        ARQ_USO.write_text(json.dumps({"dia": str(pd.Timestamp.today().date()), "n": usos_hoje() + 1}))
    except OSError:
        pass


TOKEN_MP = segredo("MERCADOPAGO_TOKEN")
PIX_ATIVO = bool(TOKEN_MP)
PRECO_PACOTE = float(segredo("PRECO_PACOTE", 10))
ANALISES_PACOTE = int(segredo("ANALISES_POR_PACOTE", 5))


def painel_codigo_pix():
    """Campo do código de acesso + compra de pacote via Pix. Retorna (chave a usar, código ativo)."""
    ss = st.session_state
    if "preencher_codigo" in ss:  # código recém-pago: preenche o campo antes de ele ser criado
        ss["codigo_acesso"] = ss.pop("preencher_codigo")
    codigo = st.text_input("Código de acesso", key="codigo_acesso", placeholder="XXXX-XXXX",
                           help=f"Recebido ao comprar um pacote de {ANALISES_PACOTE} análises via Pix.")
    chave, ativo = "", None
    if codigo:
        cod = pg.normalizar_codigo(codigo)
        try:
            restantes = pg.validar(TOKEN_MP, cod, ANALISES_PACOTE)
        except pg.ErroPagamento as e:
            st.error(str(e))
        else:
            if restantes > 0:
                chave, ativo = CHAVE_DONO, cod
                st.caption(f"✅ Código válido: **{restantes}** análise(s) restante(s).")
            else:
                st.warning("Este código já usou todas as análises. Compre um novo pacote abaixo.")

    preco = f"R$ {PRECO_PACOTE:.2f}".replace(".", ",")
    if not ativo and st.toggle(f"🛒 Comprar {ANALISES_PACOTE} análises por {preco} (Pix)"):
        cob = ss.get("cobranca")
        if cob is None:
            email = st.text_input("Seu e-mail (exigido pelo Mercado Pago)")
            if st.button("Gerar Pix", type="primary"):
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email or ""):
                    st.error("Digite um e-mail válido.")
                else:
                    try:
                        ss["cobranca"] = pg.criar_pix(TOKEN_MP, email, PRECO_PACOTE, ANALISES_PACOTE)
                        st.rerun()
                    except pg.ErroPagamento as e:
                        st.error(str(e))
        else:
            st.markdown(f"**Seu código de acesso:** `{cob.codigo}`  \n"
                        "Anote! Ele será ativado assim que o pagamento for confirmado.")
            if cob.qr_base64:
                st.image(base64.b64decode(cob.qr_base64), caption="Escaneie no app do seu banco", width=220)
            st.caption("Ou use o Pix copia e cola:")
            st.code(cob.copia_e_cola, language=None, wrap_lines=True)
            st.caption(f"Válido até {cob.expira_em:%H:%M}.")
            c1, c2 = st.columns(2)
            if c1.button("Já paguei", type="primary"):
                try:
                    status = pg.status_pagamento(TOKEN_MP, cob.pagamento_id)
                except pg.ErroPagamento as e:
                    st.error(str(e))
                else:
                    if status == "approved":
                        pg.ativar(cob.codigo, cob.pagamento_id, ANALISES_PACOTE)
                        ss["preencher_codigo"] = cob.codigo
                        del ss["cobranca"]
                        st.rerun()
                    elif status in ("pending", "in_process"):
                        st.info("Pagamento ainda não confirmado. Aguarde alguns segundos e clique de novo.")
                    else:
                        st.error(f"Pagamento não aprovado (situação: {status}). Gere um novo Pix.")
            if c2.button("Cancelar"):
                del ss["cobranca"]
                st.rerun()
    return chave, ativo


# --------------------------------------------------------------------------- #
# Barra lateral
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📈 Previsão de Ações")
    with st.form("config"):
        fonte = st.radio("Fonte dos dados", ["Yahoo Finance", "Arquivo CSV", "Demonstração"],
                         help="Use 'Demonstração' para testar sem internet (dados sintéticos).")
        ticker_in = st.text_input("Ação", value="PETR4",
                                  help="Ex.: PETR4, VALE3, ITUB4, BBAS3 (B3) ou AAPL, MSFT (EUA). "
                                       "Ações da B3 recebem '.SA' automaticamente.")
        csv = st.file_uploader("Arquivo CSV (Date, Open, High, Low, Close, Volume)", type="csv")
        anos = st.slider("Anos de histórico", 3, 20, 10)
        horizontes = st.multiselect("Prever para", list(HORIZONTES), default=[1, 5, 21], format_func=desc_h)
        st.divider()
        usar_externos = st.toggle("Incluir dados externos", value=True,
                                  help="Índices, dólar, juros, VIX, petróleo, expectativas Focus e incerteza política.")
        otimizar = st.toggle("Otimizar hiperparâmetros", value=True,
                             help="Busca aleatória com validação temporal. Mais lento, geralmente mais preciso.")
        n_iter = st.slider("Combinações testadas por modelo", 5, 40, 10,
                           help="Mais combinações = busca mais completa e mais demorada.")
        n_blocos = st.slider("Blocos de validação walk-forward", 3, 10, 5)
        enviar = st.form_submit_button("🔮 Prever", type="primary", width="stretch")

    with st.expander("🤖 Análise com IA (opcional)"):
        st.caption("Usa o Claude para pesquisar notícias, cenário político e especulações na web. "
                   + ("Use a sua chave da API da Anthropic ou um código de acesso comprado via Pix."
                      if MODO_PUBLICO and PIX_ATIVO and CHAVE_DONO else
                      "Precisa de uma chave da API da Anthropic (console.anthropic.com), cobrada por uso."))
        # A chave salva nunca é colocada no campo: o valor de um campo vai para o navegador
        chave_digitada = st.text_input(
            "Sua chave da API" if MODO_PUBLICO else "Chave da API", type="password", placeholder="sk-ant-...",
            help=None if MODO_PUBLICO else "Deixe em branco para usar a chave salva neste computador.")
        usando_dono, codigo_ativo = False, None
        if chave_digitada:
            chave = chave_digitada
        elif not MODO_PUBLICO:
            chave = CHAVE_DONO
            if CHAVE_DONO:
                st.caption("✅ Usando a chave salva neste computador.")
        else:
            chave = ""
            if CHAVE_DONO and PIX_ATIVO:
                chave, codigo_ativo = painel_codigo_pix()
            if not chave and CHAVE_DONO and SENHA_APP:
                senha = st.text_input("Ou a senha de acesso do app", type="password",
                                      help="Quem tem a senha usa a chave do dono do app, com limite diário.")
                usando_dono = senha == SENHA_APP
                chave = CHAVE_DONO if usando_dono else ""
                if senha and not usando_dono:
                    st.error("Senha incorreta.")
                elif usando_dono:
                    st.caption(f"✅ Senha correta. Análises usadas hoje: {usos_hoje()} de {LIMITE_DIARIO}.")
        modelo_ia = st.selectbox("Modelo", list(ia.MODELOS), format_func=ia.MODELOS.get)
        max_buscas = st.slider("Buscas na web por análise", 3, 15, 8)
        if not MODO_PUBLICO and chave_digitada and chave_digitada != CHAVE_DONO \
                and st.button("Lembrar chave neste computador"):
            salvar_chave(chave_digitada)
            st.success("Chave salva em .streamlit/secrets.toml (na pasta do app).")

    st.caption("⚠️ Ferramenta educacional. Previsões de mercado são altamente incertas e "
               "**não constituem recomendação de investimento**.")


# --------------------------------------------------------------------------- #
# Execução do modelo
# --------------------------------------------------------------------------- #
if enviar:
    if not horizontes:
        st.error("Escolha pelo menos um horizonte de previsão.")
        st.stop()

    avisos, ticker_real = [], None
    try:
        if fonte == "Demonstração":
            ticker = "DEMO (sintético)"
            df, externos, series = mp.gerar_demo(anos)
            avisos.append("Dados **sintéticos**: a ação de demonstração foi gerada com uma pequena "
                          "dependência do mercado, do dólar e da incerteza política, para mostrar o app funcionando. "
                          "Notícias e análise com IA ficam disponíveis apenas para ações reais.")
        else:
            if fonte == "Yahoo Finance":
                ticker = ticker_real = mp.normalizar_ticker(ticker_in)
                df = carregar_yahoo(ticker, anos)
            else:
                if csv is None:
                    st.error("Envie um arquivo CSV.")
                    st.stop()
                ticker_real = mp.normalizar_ticker(ticker_in) if ticker_in else None
                ticker = ticker_real or csv.name
                df = mp.ler_csv(csv)
            externos, series = None, []
            if usar_externos:
                externos, series, av = carregar_externos(ticker_real or "", df.index[0], df.index[-1])
                avisos += av
                if not series:
                    externos = None
                    avisos.append("Nenhum dado externo disponível — usando apenas indicadores técnicos.")
        if not usar_externos:
            externos, series = None, []
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.info("Verifique o código da ação ou use um arquivo CSV / modo Demonstração.")
        st.stop()

    barra = st.progress(0.0, text="Preparando...")
    try:
        resultados = mp.prever(
            df, externos, series, sorted(horizontes), otimizar_params=otimizar,
            n_iter=n_iter, n_blocos=n_blocos,
            progresso=lambda f, msg: barra.progress(min(f, 1.0), text=msg),
        )
    except Exception as e:
        barra.empty()
        st.error(f"Erro ao treinar os modelos: {e}")
        st.stop()
    barra.empty()

    if ticker_real:
        try:
            hist.registrar_modelo(ticker_real, df.index[-1], resultados)
        except Exception as e:
            avisos.append(f"Não foi possível salvar no histórico de previsões ({e}).")

    st.session_state["res"] = dict(ticker=ticker, ticker_real=ticker_real, df=df, externos=externos,
                                   series=series, resultados=resultados, avisos=avisos)

if "res" not in st.session_state:
    st.title("Previsão de preço de ações")
    st.markdown(
        """
Digite o código da ação na barra lateral e clique em **Prever**.

**O que o app faz**
1. **Modelo quantitativo:** aprende com cerca de 10 anos de histórico da ação e de dados externos
   (Ibovespa/S&P 500, dólar, juros, VIX, petróleo, **expectativas do mercado do boletim Focus** e
   **índice de incerteza política**). Testa tudo em validação walk-forward, como aconteceria na vida real.
2. **Notícias e contexto:** manchetes recentes com sentimento e marcação de temas políticos,
   consenso de analistas e expectativas do Focus.
3. **Análise com IA (opcional):** o Claude pesquisa na web o cenário político, o mercado e as especulações
   atuais e monta cenários otimista, base e pessimista com probabilidades.
4. **Acompanhamento:** guarda cada previsão e, quando a data chega, mostra quem acertou mais — o modelo ou a IA.
        """
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Resultados
# --------------------------------------------------------------------------- #
R = st.session_state["res"]
df, resultados, ticker_real = R["df"], R["resultados"], R["ticker_real"]
close = df["Close"]
analises = st.session_state.setdefault("analises_ia", {})
analise_atual = next((a for (t, d, h), a in analises.items() if t == ticker_real and d == df.index[-1]), None)

for a in R["avisos"]:
    st.warning(a, icon="⚠️")

st.title(R["ticker"])
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Último fechamento ({df.index[-1]:%d/%m/%Y})", fmt_preco(close.iloc[-1]),
          fmt_pct((close.iloc[-1] / close.iloc[-2] - 1) * 100))
c2.metric("Variação em 1 mês", fmt_pct((close.iloc[-1] / close.iloc[-22] - 1) * 100))
c3.metric("Variação em 12 meses", fmt_pct((close.iloc[-1] / close.iloc[-253] - 1) * 100)
          if len(close) > 253 else "—")
vol_anual = np.log(close).diff().iloc[-252:].std() * np.sqrt(252) * 100
c4.metric("Volatilidade anual", fmt_pct(vol_anual, sinal=False))

st.subheader("Previsões do modelo")
cols = st.columns(len(resultados))
icone_conf = {"Moderada": "🟢", "Baixa": "🟡", "Muito baixa": "🔴"}
for col, (h, r) in zip(cols, resultados.items()):
    with col.container(border=True):
        st.metric(f"Em {desc_h(h)} ({r.data_alvo:%d/%m/%Y})", fmt_preco(r.preco_previsto), fmt_pct(r.retorno_previsto))
        direcao = "↑ Alta" if r.retorno_previsto > 0 else "↓ Baixa"
        st.markdown(
            f"**Tendência:** {direcao}  \n"
            f"**Probabilidade de alta:** {r.prob_alta * 100:.0f}%  \n"
            f"**Faixa provável (80%):** {fmt_preco(r.intervalo_80[0])} a {fmt_preco(r.intervalo_80[1])}  \n"
            f"**Confiabilidade:** {icone_conf[r.confiabilidade]} {r.confiabilidade}  \n"
            f"<small>Modelo: {r.melhor_modelo} · acerto de direção no teste {r.acerto_direcao:.1f}%</small>",
            unsafe_allow_html=True,
        )
        vantagem = r.p_valor_sinal_forte < 0.05 and r.acerto_sinal_forte > 52
        if r.sinal_forte and vantagem:
            st.success(f"💪 **Sinal forte.** No teste, previsões fortes como esta acertaram a direção em "
                       f"**{r.acerto_sinal_forte:.0f}%** das vezes ({r.n_sinal_forte} casos).")
        elif r.sinal_forte:
            st.info(f"Sinal forte, mas no teste os sinais fortes acertaram só {r.acerto_sinal_forte:.0f}% "
                    f"— não dá para distinguir de sorte.")
        else:
            st.caption("Sinal fraco: o modelo prevê pouco movimento. "
                       + (f"Vale esperar um sinal forte (acerto de {r.acerto_sinal_forte:.0f}% no teste)."
                          if vantagem else "Neste horizonte nem os sinais fortes mostraram vantagem."))
        a = analises.get((ticker_real, df.index[-1], h))
        if a:
            st.markdown(f"🤖 **Visão da IA:** {fmt_preco(a.preco_esperado)} ({fmt_pct(a.retorno_esperado)})")

# Gráfico principal: histórico + previsões com intervalos
ultimo_ano = close.iloc[-252:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=ultimo_ano.index, y=ultimo_ano, name="Fechamento", mode="lines",
                         line=dict(color=AZUL, width=2),
                         hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f}<extra></extra>"))
rs = list(resultados.values())
datas = [df.index[-1]] + [r.data_alvo for r in rs]
for nivel, alpha in (("95", 0.12), ("80", 0.25)):
    sup = [close.iloc[-1]] + [getattr(r, f"intervalo_{nivel}")[1] for r in rs]
    inf = [close.iloc[-1]] + [getattr(r, f"intervalo_{nivel}")[0] for r in rs]
    fig.add_trace(go.Scatter(x=datas, y=sup, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=datas, y=inf, mode="lines", fill="tonexty", fillcolor=f"rgba(235,104,52,{alpha})",
                             line=dict(width=0), name=f"Intervalo {nivel}%", hoverinfo="skip"))
fig.add_trace(go.Scatter(
    x=datas, y=[close.iloc[-1]] + [r.preco_previsto for r in rs], name="Previsão do modelo",
    mode="lines+markers", line=dict(color=LARANJA, width=2, dash="dot"), marker=dict(size=9),
    customdata=[[0, 0, 0]] + [[r.retorno_previsto, r.intervalo_80[0], r.intervalo_80[1]] for r in rs],
    hovertemplate="%{x|%d/%m/%Y}<br>Modelo: %{y:.2f} (%{customdata[0]:+.2f}%)"
                  "<br>Faixa 80%: %{customdata[1]:.2f} a %{customdata[2]:.2f}<extra></extra>"))
pontos_ia = [(resultados[h].data_alvo, a) for (t, d, h), a in analises.items()
             if t == ticker_real and d == df.index[-1] and h in resultados]
if pontos_ia:
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pontos_ia], y=[p[1].preco_esperado for p in pontos_ia], name="Visão da IA",
        mode="markers", marker=dict(size=13, symbol="diamond", color=VIOLETA, line=dict(width=2, color="white")),
        hovertemplate="%{x|%d/%m/%Y}<br>IA (cenários ponderados): %{y:.2f}<extra></extra>"))
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  legend=dict(orientation="h", y=1.08, x=0), yaxis_title="Preço")
st.plotly_chart(fig, width="stretch")

aba_modelo, aba_ctx, aba_ia, aba_acomp = st.tabs(
    ["📊 Modelo", "📰 Notícias e contexto", "🤖 Análise com IA", "🎯 Acompanhamento"])


# --------------------------------------------------------------------------- #
# Aba: Modelo
# --------------------------------------------------------------------------- #
with aba_modelo:
    h_sel = st.segmented_control("Horizonte", list(resultados), format_func=desc_h,
                                 default=list(resultados)[0]) or list(resultados)[0]
    r = resultados[h_sel]
    sub1, sub2, sub3, sub4 = st.tabs(["Desempenho", "Indicadores", "Dados externos", "Parâmetros"])

    with sub1:
        st.markdown(
            f"Teste **fora da amostra** em {len(r.previsoes_oos)} pregões (os últimos ~40% do histórico). "
            f"O melhor modelo (**{r.melhor_modelo}**) teve erro "
            f"{'**menor**' if r.ganho_vs_baseline > 0 else '**maior**'} que simplesmente prever "
            f"\"preço não muda\" ({fmt_pct(r.ganho_vs_baseline)} no RMSE)."
        )
        if r.ganho_vs_baseline <= 0:
            st.error("O modelo não superou o baseline ingênuo neste horizonte. Trate a previsão com muito ceticismo.")
        st.markdown(
            f"**Sinais fortes** (as 25% previsões mais intensas): acerto de direção de "
            f"**{r.acerto_sinal_forte:.1f}%** em {r.n_sinal_forte} casos, contra {r.acerto_direcao:.1f}% no geral. "
            + ("Isso é estatisticamente melhor que jogar uma moeda "
               f"(chance de ser sorte: {r.p_valor_sinal_forte * 100:.1f}%)."
               if r.p_valor_sinal_forte < 0.05 else
               f"Não dá para descartar sorte (chance de ser sorte: {r.p_valor_sinal_forte * 100:.0f}%).")
        )
        st.dataframe(r.metricas.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"), width="stretch")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Backtest: comprado quando o modelo prevê alta**")
            st.caption("Sem custos de corretagem e impostos. Operações a cada horizonte, sem sobreposição.")
            curvas = r.curvas
            fig_bt = go.Figure()
            for nome, cor, dash in [(r.melhor_modelo, LARANJA, None), ("Comprar e segurar", AZUL, None),
                                    ("Baseline: tendência 20d", VERDE_AGUA, "dot")]:
                fig_bt.add_trace(go.Scatter(x=curvas.index, y=(curvas[nome] - 1) * 100, name=nome,
                                            line=dict(color=cor, width=2, dash=dash),
                                            hovertemplate="%{y:.1f}%<extra>" + nome + "</extra>"))
            fig_bt.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                                 yaxis_title="Retorno acumulado (%)", legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_bt, width="stretch")
        with g2:
            st.markdown("**Preço previsto x preço real (fora da amostra)**")
            st.caption(f"Cada ponto é a previsão feita {h_sel} pregão(ões) antes.")
            prev = r.previsoes_oos.iloc[-252:]
            base = close.reindex(prev.index)
            alvo = close.shift(-h_sel).reindex(prev.index)
            datas_alvo = close.index.to_series().shift(-h_sel).reindex(prev.index)
            fig_oos = go.Figure()
            fig_oos.add_trace(go.Scatter(x=datas_alvo, y=alvo, name="Real", line=dict(color=AZUL, width=2)))
            fig_oos.add_trace(go.Scatter(x=datas_alvo, y=base * np.exp(prev[r.melhor_modelo]), name="Previsto",
                                         line=dict(color=LARANJA, width=1.5)))
            fig_oos.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                                  legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig_oos, width="stretch")

    with sub2:
        nomes_ext = {s.nome for s in R["series"]}
        externos_cols = {"forca_rel_20d", "forca_rel_60d", "beta_60d", "correl_60d"}

        def eh_externo(c: str) -> bool:
            return c in externos_cols or any(c.startswith(n + "_") for n in nomes_ext)

        top = r.importancias.head(20).iloc[::-1]
        grupo = ["Externo" if eh_externo(c) else "Técnico" for c in top.index]
        fig_imp = go.Figure()
        for g, cor in [("Técnico", AZUL), ("Externo", LARANJA)]:
            mask = [x == g for x in grupo]
            if any(mask):
                fig_imp.add_trace(go.Bar(x=top[mask] * 100, y=top.index[mask], orientation="h", name=g,
                                         marker_color=cor, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig_imp.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Importância (%)",
                              yaxis=dict(categoryorder="array", categoryarray=list(top.index)),
                              legend=dict(orientation="h", y=1.05))
        st.markdown(f"**20 indicadores mais relevantes** (de {r.n_features}), segundo o Random Forest")
        st.plotly_chart(fig_imp, width="stretch")
        peso_ext = sum(v for c, v in r.importancias.items() if eh_externo(c)) * 100
        st.caption(f"Peso total dos dados externos no modelo: {peso_ext:.0f}%")

    with sub3:
        if R["externos"] is None or not R["series"]:
            st.info("Dados externos não foram usados nesta análise.")
        else:
            ext = mp.alinhar(R["externos"], df.index, R["series"]).iloc[-252:]
            ret_acao = np.log(close).diff().iloc[-252:]
            linhas = []
            for s in R["series"]:
                x = ext[s.nome]
                var = np.log(x).diff() if s.tipo == "preco" else x.diff()
                linhas.append({
                    "Série": s.descricao,
                    "Último valor": x.iloc[-1],
                    "Variação 1 mês": (x.iloc[-1] / x.iloc[-22] - 1) * 100 if s.tipo == "preco"
                    else x.iloc[-1] - x.iloc[-22],
                    "Correlação diária com a ação (12m)": ret_acao.corr(var),
                })
            st.dataframe(pd.DataFrame(linhas).set_index("Série").map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
                         width="stretch")
            st.caption("Para juros, inflação esperada, VIX e incerteza, a variação de 1 mês está em pontos; "
                       "para os demais, em %.")

            escolha = st.selectbox("Comparar com", [s.descricao for s in R["series"]])
            s = next(s for s in R["series"] if s.descricao == escolha)
            base100 = pd.DataFrame({R["ticker"]: close.iloc[-252:] / close.iloc[-252] * 100,
                                    escolha: ext[s.nome] / ext[s.nome].iloc[0] * 100})
            fig_cmp = go.Figure()
            for nome, cor in zip(base100.columns, [AZUL, LARANJA]):
                fig_cmp.add_trace(go.Scatter(x=base100.index, y=base100[nome], name=nome,
                                             line=dict(color=cor, width=2)))
            fig_cmp.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                                  yaxis_title="Base 100 (12 meses atrás)", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig_cmp, width="stretch")

    with sub4:
        st.markdown(f"Treino com **{r.n_amostras}** pregões e **{r.n_features}** indicadores.")
        for nome, params in r.parametros.items():
            st.markdown(f"**{nome}**")
            st.json({k: (v.item() if isinstance(v, (np.floating, np.integer)) else v) for k, v in params.items()})


# --------------------------------------------------------------------------- #
# Aba: Notícias e contexto
# --------------------------------------------------------------------------- #
ctx = None
with aba_ctx:
    if not ticker_real:
        st.info("Notícias e contexto estão disponíveis apenas para ações reais (Yahoo Finance ou CSV com o código da ação).")
    else:
        ctx = carregar_contexto(ticker_real)
        for a in ctx["avisos"]:
            st.warning(a, icon="⚠️")

        emp = ctx["empresa"]
        if emp.get("nome"):
            st.markdown(f"### {emp['nome']}")
            st.caption(" · ".join(x for x in (emp.get("setor"), emp.get("industria")) if x))

        todas = ctx["noticias_empresa"] + ctx["noticias_mercado"]
        sent_emp = cx.resumo_sentimento(ctx["noticias_empresa"])
        sent_merc = cx.resumo_sentimento(ctx["noticias_mercado"])

        def rotulo_sent(v: float) -> str:
            return "🟢 Positivo" if v > 0.15 else "🔴 Negativo" if v < -0.15 else "⚪ Neutro"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Notícias da empresa (7 dias)", sent_emp["n"])
        k2.metric("Tom das notícias da empresa", rotulo_sent(sent_emp["media"]),
                  f"{sent_emp['positivas']} positivas · {sent_emp['negativas']} negativas", delta_color="off")
        k3.metric("Tom do mercado (3 dias)", rotulo_sent(sent_merc["media"]),
                  f"{sent_merc['positivas']} positivas · {sent_merc['negativas']} negativas", delta_color="off")
        k4.metric("Manchetes com tema político", f"{sum(n.politica for n in todas)} de {len(todas)}")
        st.caption("O tom é estimado por palavras-chave nas manchetes (ex.: \"dispara\", \"queda\", \"rebaixa\"). "
                   "É uma leitura rápida, não uma análise — para isso, use a aba **Análise com IA**.")

        if emp.get("preco_alvo_medio"):
            st.markdown("#### Consenso de analistas")
            a1, a2, a3 = st.columns(3)
            alvo = emp["preco_alvo_medio"]
            a1.metric("Preço-alvo médio", fmt_preco(alvo), fmt_pct((alvo / close.iloc[-1] - 1) * 100))
            a2.metric("Faixa dos preços-alvo",
                      f"{fmt_preco(emp['preco_alvo_min'])} – {fmt_preco(emp['preco_alvo_max'])}"
                      if emp.get("preco_alvo_min") else "—")
            a3.metric("Recomendação média", cx.RECOMENDACOES.get(emp.get("recomendacao"), emp.get("recomendacao") or "—"),
                      f"{emp.get('n_analistas') or '?'} analistas", delta_color="off")

        if len(ctx["focus"]):
            st.markdown("#### Expectativas do mercado (boletim Focus)")
            f = ctx["focus"].copy()
            f["Data"] = f["Data"].dt.strftime("%d/%m/%Y")
            for c, fmt in (("Mediana", "{:.2f}"), ("Há 4 semanas", "{:.2f}"), ("Mudança", "{:+.2f}")):
                f[c] = f[c].map(lambda v, fmt=fmt: "—" if pd.isna(v) else fmt.format(v))
            st.dataframe(f.rename(columns={"Data": "Atualizado em"}), hide_index=True, width="stretch")
            st.caption("Mediana das projeções de economistas coletadas pelo Banco Central. "
                       "Mudança = variação em relação a ~4 semanas atrás.")

        def lista_noticias(noticias):
            if not noticias:
                st.caption("Nenhuma notícia encontrada.")
            for n in noticias:
                icone = "🟢" if n.sentimento > 0 else "🔴" if n.sentimento < 0 else "⚪"
                tag = " 🏛️" if n.politica else ""
                data = f"{n.data:%d/%m %H:%M}" if n.data is not None else ""
                st.markdown(f"{icone}{tag} [{n.titulo}]({n.link})  \n<small>{n.fonte} · {data}</small>",
                            unsafe_allow_html=True)

        n1, n2 = st.columns(2)
        with n1:
            st.markdown("#### Notícias da empresa")
            lista_noticias(ctx["noticias_empresa"][:20])
        with n2:
            st.markdown("#### Mercado e política")
            lista_noticias(ctx["noticias_mercado"][:20])
        st.caption("🟢 tom positivo · 🔴 tom negativo · ⚪ neutro · 🏛️ tema político")


# --------------------------------------------------------------------------- #
# Aba: Análise com IA
# --------------------------------------------------------------------------- #
ROTULO_SENT = {"muito_negativo": "🔴🔴 Muito negativo", "negativo": "🔴 Negativo", "neutro": "⚪ Neutro",
               "positivo": "🟢 Positivo", "muito_positivo": "🟢🟢 Muito positivo"}
ROTULO_CAT = {"politico": "🏛️ Político", "macroeconomico": "🌎 Macroeconomia", "setorial": "🏭 Setor",
              "empresa": "🏢 Empresa", "especulacao": "💬 Especulação"}
ROTULO_CONC = {"concorda": "✅ Concorda com o modelo", "discorda_parcialmente": "➖ Discorda em parte do modelo",
               "discorda": "❌ Discorda do modelo"}

with aba_ia:
    if not ticker_real:
        st.info("A análise com IA está disponível apenas para ações reais.")
    else:
        st.markdown(
            "O Claude pesquisa na web as notícias mais recentes sobre a **empresa**, o **cenário político**, a "
            "**macroeconomia** e as **especulações do mercado**, junta com o resultado do modelo quantitativo e "
            "monta cenários com probabilidades."
        )
        h_ia = st.selectbox("Horizonte dos cenários", list(resultados), index=len(resultados) - 1, format_func=desc_h)
        st.caption(f"Custo estimado por análise: cerca de US$ 0,15 a 0,60 com o Opus e "
                   f"metade disso com o Sonnet, dependendo do número de buscas.")
        limite_atingido = usando_dono and usos_hoje() >= LIMITE_DIARIO
        if limite_atingido:
            st.warning(f"O limite de {LIMITE_DIARIO} análises por dia com a chave do app foi atingido. "
                       "Volte amanhã ou use a sua própria chave da API.")
        if st.button("🤖 Gerar análise com IA", type="primary", disabled=not chave or limite_atingido):
            if usando_dono:
                registrar_uso()
            if codigo_ativo:
                try:
                    pg.usar(codigo_ativo)
                except pg.ErroPagamento as e:
                    st.error(str(e))
                    st.stop()
            if ctx is None:
                ctx = carregar_contexto(ticker_real)
            texto = ia.montar_contexto_texto(ticker_real, df, resultados, ctx, HORIZONTES)
            with st.status("Analisando...", expanded=True) as status:
                try:
                    analise = ia.analisar(texto, ticker_real, float(close.iloc[-1]),
                                          f"{desc_h(h_ia)} (até {resultados[h_ia].data_alvo:%d/%m/%Y})",
                                          chave_api=chave, modelo=modelo_ia, max_buscas=max_buscas,
                                          progresso=st.write)
                except ia.ErroAnalise as e:
                    if codigo_ativo:
                        pg.devolver(codigo_ativo)  # análise falhou: o crédito volta
                    status.update(label="Falhou", state="error")
                    st.error(str(e))
                    st.stop()
                status.update(label="Análise concluída", state="complete")
            analises[(ticker_real, df.index[-1], h_ia)] = analise
            try:
                hist.registrar_ia(ticker_real, df.index[-1], h_ia, analise.preco_esperado, modelo_ia)
            except Exception:
                pass
            st.rerun()
        if not chave:
            opcoes = ["a sua chave da API"]
            if MODO_PUBLICO and PIX_ATIVO and CHAVE_DONO:
                opcoes.append(f"um código de acesso (compre {ANALISES_PACOTE} análises via Pix)")
            if MODO_PUBLICO and SENHA_APP and CHAVE_DONO:
                opcoes.append("a senha de acesso")
            st.info("Para usar, informe " + " ou ".join(opcoes) + " na barra lateral, em **🤖 Análise com IA**.")

        a = analises.get((ticker_real, df.index[-1], h_ia))
        if a:
            d = a.dados
            r = resultados[h_ia]
            st.divider()
            st.markdown(f"### Leitura geral — {ROTULO_SENT[d['sentimento_geral']]}")
            st.write(d["resumo"])
            st.caption(f"Confiança da IA: **{d['confianca']}** · {ROTULO_CONC[d['concordancia_com_modelo']]}")

            st.markdown(f"#### Cenários para {desc_h(h_ia)}")
            ordem = {"otimista": 0, "base": 1, "pessimista": 2}
            icones = {"otimista": "📈", "base": "➡️", "pessimista": "📉"}
            cc = st.columns(3)
            for col, c in zip(cc, sorted(d["cenarios"], key=lambda c: ordem[c["nome"]])):
                with col.container(border=True):
                    st.metric(f"{icones[c['nome']]} {c['nome'].capitalize()} · {c['probabilidade']:.0f}%",
                              fmt_preco(c["preco_alvo"]), fmt_pct((c["preco_alvo"] / close.iloc[-1] - 1) * 100))
                    st.caption(c["premissas"])

            st.markdown("#### Comparação")
            comp = [
                {"Fonte": "Modelo quantitativo", "Preço": r.preco_previsto, "Variação (%)": r.retorno_previsto},
                {"Fonte": "IA (média ponderada dos cenários)", "Preço": a.preco_esperado,
                 "Variação (%)": a.retorno_esperado},
            ]
            emp = (ctx or {}).get("empresa") or {}
            if emp.get("preco_alvo_medio"):
                comp.append({"Fonte": "Consenso de analistas (12 meses)", "Preço": emp["preco_alvo_medio"],
                             "Variação (%)": (emp["preco_alvo_medio"] / close.iloc[-1] - 1) * 100})
            comp = pd.DataFrame(comp).set_index("Fonte")
            comp["Preço"] = comp["Preço"].map(fmt_preco)
            comp["Variação (%)"] = comp["Variação (%)"].map(fmt_pct)
            st.dataframe(comp, width="stretch")
            st.caption(d["comentario_modelo"])

            st.markdown("#### Fatores que podem mover a ação")
            for cat, rot in ROTULO_CAT.items():
                fat = [f for f in d["fatores"] if f["categoria"] == cat]
                if not fat:
                    continue
                st.markdown(f"**{rot}**")
                for f in fat:
                    seta = {"alta": "🟢 ↑", "baixa": "🔴 ↓", "neutro": "⚪ →"}[f["impacto"]]
                    rumor = " · _rumor, não confirmado_" if not f["confirmado"] else ""
                    st.markdown(f"- {seta} {f['descricao']} <small>({f['intensidade']}{rumor} · {f['fonte']})</small>",
                                unsafe_allow_html=True)

            if d["catalisadores"]:
                st.markdown("#### Próximos eventos")
                st.dataframe(pd.DataFrame(d["catalisadores"]).rename(
                    columns={"evento": "Evento", "data": "Quando", "por_que_importa": "Por que importa"}),
                    hide_index=True, width="stretch")
            if d["riscos_principais"]:
                st.markdown("#### Principais riscos")
                st.markdown("\n".join(f"- {x}" for x in d["riscos_principais"]))

            with st.expander(f"Fontes consultadas ({len(a.fontes)})"):
                for f in a.fontes:
                    st.markdown(f"- [{f['titulo']}]({f['url']})" + (f" · {f['data']}" if f.get("data") else ""))
            with st.expander("Pesquisa completa da IA"):
                st.markdown(a.pesquisa)
            st.caption(f"Modelo: {ia.MODELOS[a.modelo]} · {a.buscas} buscas · custo aproximado US$ {a.custo_usd:.2f}. "
                       "A análise com IA é qualitativa e não foi validada com dados passados — "
                       "acompanhe a taxa de acerto na aba 🎯 Acompanhamento.")


# --------------------------------------------------------------------------- #
# Aba: Acompanhamento
# --------------------------------------------------------------------------- #
with aba_acomp:
    st.markdown(
        "Cada previsão feita com uma ação real fica salva em `historico_previsoes.csv`, na pasta do app. "
        "Quando a data-alvo passa, o app compara com o preço real. **É assim que você descobre, com o tempo, "
        "se o modelo e a análise com IA realmente acertam.**"
    )
    if MODO_PUBLICO:
        st.caption("No app publicado, o histórico é compartilhado entre todos os visitantes e pode ser apagado "
                   "quando o servidor reinicia. Para um histórico só seu, rode o app no seu computador.")
    h_all = hist.carregar()
    if h_all.empty:
        st.info("Nenhuma previsão salva ainda.")
    else:
        precos = {}
        for t in h_all["ticker"].dropna().unique():
            if t == ticker_real:
                precos[t] = close
            else:
                try:
                    precos[t] = carregar_yahoo(t, 1)["Close"]
                except Exception:
                    pass
        av = hist.avaliar(h_all, precos)
        pl = hist.placar(av)
        if pl.empty:
            st.info(f"{len(av)} previsões salvas. Nenhuma venceu ainda — volte depois da primeira data-alvo.")
        else:
            st.markdown("#### Placar")
            st.dataframe(pl.set_index(["Fonte", "Horizonte"]).map(lambda v: f"{v:.1f}" if isinstance(v, float) else v),
                         width="stretch")
            st.caption("Acerto de direção acima de 50% de forma consistente, com dezenas de previsões, "
                       "é o que indica alguma vantagem. Poucas previsões ainda dizem pouco.")
        st.markdown("#### Previsões salvas")
        mostrar = av.sort_values("data_base", ascending=False)[[
            "data_base", "ticker", "horizonte", "data_alvo", "preco_base", "previsto_modelo", "previsto_ia",
            "preco_real", "acertou_direcao_modelo", "acertou_direcao_ia"]].copy()
        for c in ("data_base", "data_alvo"):
            mostrar[c] = mostrar[c].dt.strftime("%d/%m/%Y")
        for c in ("acertou_direcao_modelo", "acertou_direcao_ia"):
            mostrar[c] = mostrar[c].map(lambda v: "✅" if v == 1 else "❌" if v == 0 else "—")
        for c in ("preco_base", "previsto_modelo", "previsto_ia", "preco_real"):
            mostrar[c] = mostrar[c].map(lambda v: "—" if pd.isna(v) else fmt_preco(v))
        st.dataframe(mostrar.rename(columns={
            "data_base": "Data", "ticker": "Ação", "horizonte": "Horizonte", "data_alvo": "Data-alvo",
            "preco_base": "Preço na data", "previsto_modelo": "Modelo", "previsto_ia": "IA",
            "preco_real": "Real", "acertou_direcao_modelo": "Modelo acertou?", "acertou_direcao_ia": "IA acertou?",
        }), hide_index=True, width="stretch")
