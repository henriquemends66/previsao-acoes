"""
Contexto atual da ação: notícias (com sentimento e marcação de temas políticos),
consenso de analistas e expectativas do mercado (boletim Focus do Banco Central).

Diferente do motor de previsão, estes dados NÃO têm histórico gratuito longo o suficiente
para treinar e validar um modelo — por isso são mostrados como contexto e enviados ao
analista de IA, e não entram no modelo quantitativo.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import pandas as pd

from motor_previsao import erro_curto, http_get


# --------------------------------------------------------------------------- #
# Notícias (Google News RSS — gratuito, sem chave)
# --------------------------------------------------------------------------- #
@dataclass
class Noticia:
    titulo: str
    fonte: str
    data: pd.Timestamp | None
    link: str
    sentimento: float = 0.0   # -1 (negativo) a +1 (positivo)
    politica: bool = False


def _sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", txt.lower()) if unicodedata.category(c) != "Mn")


# Léxico financeiro simples em português (sem acentos). Termos com "*" casam por prefixo
# (ex.: "dispar*" -> dispara/disparam); os demais só como palavra inteira ("cai" não casa com "Caixa").
POSITIVAS = [
    "alta", "altas", "sobe", "subiu", "sobem", "dispar*", "avanc*", "lucro*", "recorde*", "valoriz*",
    "ganho*", "otimis*", "supera*", "superou", "cresc*", "aprova*", "dividendo*", "upgrade", "eleva*",
    "compra", "positiv*", "recuper*", "forte*", "salta*", "saltou", "melhora*", "acima do esperado",
    "bate", "expans*", "acordo", "alivio",
]
NEGATIVAS = [
    "queda*", "cai", "caiu", "caem", "despenc*", "recua*", "recuo", "prejuizo*", "perda*", "rebaix*",
    "downgrade", "pessimis*", "crise", "investiga*", "multa*", "greve", "fraude*", "deficit", "corte*",
    "desaba*", "tomba*", "fraco*", "fraca*", "incerteza*", "rombo", "intervenc*", "preocupa*", "risco*",
    "abaixo do esperado", "piora*", "tensao", "sanc*", "tarifa*", "processo*", "calote", "rejeita*",
    "derrota", "escandalo",
]
TEMAS_POLITICOS = [
    "governo*", "lula", "congresso", "senado", "camara", "stf", "eleic*", "eleitora*", "fiscal",
    "arcabouco", "haddad", "ministro*", "ministerio", "reforma*", "tributar*", "planalto", "presidente",
    "copom", "banco central", "galipolo", "orcamento", "intervenc*", "estatal", "estatais", "privatiz*",
    "pec", "trump", "tarifa*", "sanc*", "bolsonaro", "tarcisio", "oposicao", "votac*",
]


def _conta(texto: str, termos: list[str]) -> int:
    total = 0
    for t in termos:
        padrao = r"\b" + re.escape(t.rstrip("*")) + (r"" if t.endswith("*") else r"\b")
        total += bool(re.search(padrao, texto))
    return total


def sentimento_titulo(titulo: str) -> float:
    t = _sem_acento(titulo)
    pos, neg = _conta(t, POSITIVAS), _conta(t, NEGATIVAS)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def eh_politica(titulo: str) -> bool:
    return _conta(_sem_acento(titulo), TEMAS_POLITICOS) > 0


def parse_rss(xml: str) -> list[Noticia]:
    raiz = ET.fromstring(xml)
    noticias = []
    for item in raiz.iter("item"):
        titulo = (item.findtext("title") or "").strip()
        fonte = (item.findtext("source") or "").strip()
        # O Google News termina o título com " - Fonte"
        if fonte and titulo.endswith(f" - {fonte}"):
            titulo = titulo[: -len(fonte) - 3]
        try:
            data = pd.Timestamp(parsedate_to_datetime(item.findtext("pubDate"))).tz_convert(None)
        except Exception:
            data = None
        noticias.append(Noticia(titulo, fonte, data, item.findtext("link") or "",
                                sentimento_titulo(titulo), eh_politica(titulo)))
    return noticias


def buscar_noticias(consulta: str, dias: int = 7, limite: int = 40) -> list[Noticia]:
    url = "https://news.google.com/rss/search"
    params = {"q": f"{consulta} when:{dias}d", "hl": "pt-BR", "gl": "BR", "ceid": "BR:pt-419"}
    noticias = parse_rss(http_get(url, params).text)
    # Remove títulos repetidos e ordena do mais recente para o mais antigo
    vistos, unicas = set(), []
    for n in noticias:
        chave = _sem_acento(n.titulo)[:80]
        if chave not in vistos:
            vistos.add(chave)
            unicas.append(n)
    unicas.sort(key=lambda n: n.data or pd.Timestamp.min, reverse=True)
    return unicas[:limite]


def resumo_sentimento(noticias: list[Noticia]) -> dict:
    if not noticias:
        return {"n": 0, "media": 0.0, "positivas": 0, "negativas": 0, "politicas": 0}
    s = pd.Series([n.sentimento for n in noticias])
    return {
        "n": len(noticias),
        "media": float(s.mean()),
        "positivas": int((s > 0).sum()),
        "negativas": int((s < 0).sum()),
        "politicas": sum(n.politica for n in noticias),
    }


# --------------------------------------------------------------------------- #
# Dados da empresa e consenso de analistas (Yahoo Finance)
# --------------------------------------------------------------------------- #
def info_empresa(ticker: str) -> dict:
    """Nome, setor e consenso de analistas. Campos ausentes ficam como None."""
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    return {
        "nome": info.get("longName") or info.get("shortName"),
        "nome_curto": info.get("shortName"),
        "setor": info.get("sector"),
        "industria": info.get("industry"),
        "preco_alvo_medio": info.get("targetMeanPrice"),
        "preco_alvo_max": info.get("targetHighPrice"),
        "preco_alvo_min": info.get("targetLowPrice"),
        "recomendacao": info.get("recommendationKey"),
        "n_analistas": info.get("numberOfAnalystOpinions"),
        "pl": info.get("trailingPE"),
        "dividend_yield": info.get("dividendYield"),
    }


RECOMENDACOES = {
    "strong_buy": "Compra forte", "buy": "Compra", "hold": "Neutro",
    "underperform": "Desempenho abaixo do mercado", "sell": "Venda", "strong_sell": "Venda forte",
}


# --------------------------------------------------------------------------- #
# Boletim Focus (expectativas do mercado — Banco Central)
# --------------------------------------------------------------------------- #
INDICADORES_FOCUS = ["IPCA", "Selic", "Câmbio", "PIB Total"]


def focus_anual(hoje: pd.Timestamp | None = None) -> pd.DataFrame:
    """Mediana das expectativas para este ano e o próximo, e a mudança em ~4 semanas."""
    hoje = hoje or pd.Timestamp.today().normalize()
    url = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
           "ExpectativasMercadoAnuais")
    indicadores = " or ".join(f"Indicador eq '{i}'" for i in INDICADORES_FOCUS)
    params = {
        "$filter": f"({indicadores}) and baseCalculo eq 0 and Data ge '{hoje - pd.Timedelta(days=45):%Y-%m-%d}'",
        "$select": "Indicador,Data,DataReferencia,Mediana",
        "$top": "5000", "$format": "json",
    }
    return tabela_focus(pd.DataFrame(http_get(url, params).json()["value"]), hoje.year)


def tabela_focus(dados: pd.DataFrame, ano: int) -> pd.DataFrame:
    dados = dados.copy()
    dados["Data"] = pd.to_datetime(dados["Data"])
    dados["DataReferencia"] = dados["DataReferencia"].astype(int)
    linhas = []
    for ind in INDICADORES_FOCUS:
        for ref in (ano, ano + 1):
            d = dados[(dados["Indicador"] == ind) & (dados["DataReferencia"] == ref)].sort_values("Data")
            if d.empty:
                continue
            atual = d.iloc[-1]
            antes = d[d["Data"] <= atual["Data"] - pd.Timedelta(days=28)]
            linhas.append({
                "Indicador": ind, "Ano": ref, "Mediana": float(atual["Mediana"]),
                "Há 4 semanas": float(antes.iloc[-1]["Mediana"]) if len(antes) else None,
                "Data": atual["Data"],
            })
    tab = pd.DataFrame(linhas)
    if not tab.empty:
        tab["Mudança"] = tab["Mediana"] - tab["Há 4 semanas"]
    return tab


# --------------------------------------------------------------------------- #
# Tudo junto
# --------------------------------------------------------------------------- #
def coletar(ticker: str) -> dict:
    """Coleta o contexto; cada parte falha de forma independente e gera um aviso."""
    ctx = {"empresa": {}, "noticias_empresa": [], "noticias_mercado": [], "focus": pd.DataFrame(), "avisos": []}
    base = ticker.replace(".SA", "")
    try:
        ctx["empresa"] = info_empresa(ticker)
    except Exception as e:
        ctx["avisos"].append(f"Consenso de analistas indisponível ({erro_curto(e)}).")

    nome = ctx["empresa"].get("nome_curto") or ctx["empresa"].get("nome")
    consulta = f'"{base}"' + (f' OR "{nome.split()[0].title()}"' if nome else "")
    try:
        ctx["noticias_empresa"] = buscar_noticias(consulta)
    except Exception as e:
        ctx["avisos"].append(f"Notícias da empresa indisponíveis ({erro_curto(e)}).")

    brasil = ticker.upper().endswith(".SA")
    consulta_mercado = ("Ibovespa OR mercado financeiro OR Copom OR fiscal OR governo economia"
                        if brasil else "stock market OR Fed OR Wall Street")
    try:
        ctx["noticias_mercado"] = buscar_noticias(consulta_mercado, dias=3)
    except Exception as e:
        ctx["avisos"].append(f"Notícias de mercado indisponíveis ({erro_curto(e)}).")

    if brasil:
        try:
            ctx["focus"] = focus_anual()
        except Exception as e:
            ctx["avisos"].append(f"Boletim Focus indisponível ({erro_curto(e)}).")
    return ctx
