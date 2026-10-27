"""
Histórico de previsões: salva cada previsão feita no app e, quando a data-alvo passa,
compara com o preço real. É a única forma honesta de saber se o modelo e a análise
de IA realmente acertam — o teste com dados passados não mede a análise de notícias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path(__file__).with_name("historico_previsoes.csv")
COLUNAS = [
    "data_base", "ticker", "horizonte", "data_alvo", "preco_base",
    "previsto_modelo", "prob_alta", "confiabilidade", "previsto_ia", "modelo_ia", "registrado_em",
]


def carregar(arquivo: Path = ARQUIVO) -> pd.DataFrame:
    if not arquivo.exists():
        return pd.DataFrame(columns=COLUNAS)
    h = pd.read_csv(arquivo, parse_dates=["data_base", "data_alvo", "registrado_em"])
    for c in COLUNAS:
        if c not in h.columns:
            h[c] = np.nan
    for c in ("ticker", "confiabilidade", "modelo_ia"):  # colunas de texto podem vir vazias (float)
        h[c] = h[c].astype(object)
    return h[COLUNAS]


def _salvar(h: pd.DataFrame, arquivo: Path):
    h.sort_values(["data_base", "ticker", "horizonte"]).to_csv(arquivo, index=False)


def registrar_modelo(ticker: str, data_base, resultados: dict, arquivo: Path = ARQUIVO):
    """Grava (ou atualiza) as previsões do modelo. Uma linha por ação, data-base e horizonte."""
    h = carregar(arquivo)
    agora = pd.Timestamp.now().floor("s")
    for hz, r in resultados.items():
        chave = (h["ticker"] == ticker) & (h["data_base"] == pd.Timestamp(data_base)) & (h["horizonte"] == hz)
        valores = {
            "data_base": pd.Timestamp(data_base), "ticker": ticker, "horizonte": hz,
            "data_alvo": r.data_alvo, "preco_base": r.preco_atual, "previsto_modelo": r.preco_previsto,
            "prob_alta": r.prob_alta, "confiabilidade": r.confiabilidade, "registrado_em": agora,
        }
        if chave.any():
            for k, v in valores.items():
                h.loc[chave, k] = v
        else:
            h = pd.concat([h, pd.DataFrame([{**{c: np.nan for c in COLUNAS}, **valores}])], ignore_index=True)
    _salvar(h, arquivo)


def registrar_ia(ticker: str, data_base, horizonte: int, preco_ia: float, modelo_ia: str,
                 arquivo: Path = ARQUIVO):
    h = carregar(arquivo)
    chave = (h["ticker"] == ticker) & (h["data_base"] == pd.Timestamp(data_base)) & (h["horizonte"] == horizonte)
    if chave.any():
        h.loc[chave, "previsto_ia"] = preco_ia
        h.loc[chave, "modelo_ia"] = modelo_ia
        _salvar(h, arquivo)


def avaliar(h: pd.DataFrame, precos: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Preenche o preço real para previsões vencidas. `precos` mapeia ticker -> série de fechamentos.
    Usa o fechamento da data-alvo ou, se não houve pregão, o primeiro pregão seguinte.
    """
    h = h.copy()
    h["preco_real"] = np.nan
    for i, linha in h.iterrows():
        serie = precos.get(linha["ticker"])
        if serie is None or pd.isna(linha["data_alvo"]):
            continue
        depois = serie[serie.index >= linha["data_alvo"]]
        if len(depois):
            h.at[i, "preco_real"] = depois.iloc[0]

    real = h["preco_real"] / h["preco_base"] - 1
    for fonte in ("modelo", "ia"):
        prev = h[f"previsto_{fonte}"] / h["preco_base"] - 1
        h[f"erro_{fonte}_%"] = (h[f"previsto_{fonte}"] / h["preco_real"] - 1) * 100
        h[f"acertou_direcao_{fonte}"] = np.where(
            h["preco_real"].notna() & prev.notna(), np.sign(prev) == np.sign(real), np.nan)
    h["retorno_real_%"] = real * 100
    return h


def placar(avaliado: pd.DataFrame) -> pd.DataFrame:
    """Resumo da taxa de acerto por fonte (modelo x IA) e por horizonte."""
    vencidas = avaliado[avaliado["preco_real"].notna()]
    linhas = []
    for fonte, nome in (("modelo", "Modelo quantitativo"), ("ia", "Análise com IA")):
        v = vencidas[vencidas[f"previsto_{fonte}"].notna()]
        if v.empty:
            continue
        for hz, g in [("Todos", v)] + list(v.groupby("horizonte")):
            linhas.append({
                "Fonte": nome, "Horizonte": hz, "Previsões avaliadas": len(g),
                "Acerto de direção (%)": g[f"acertou_direcao_{fonte}"].astype(float).mean() * 100,
                "Erro médio absoluto (%)": g[f"erro_{fonte}_%"].abs().mean(),
            })
    return pd.DataFrame(linhas)
