"""
Motor de previsão de preço de ações.

- Dados da ação: Yahoo Finance, CSV ou série sintética de demonstração.
- Dados externos: índice de mercado, dólar, S&P 500, VIX, petróleo, juros (Selic via Banco Central / Treasury 10 anos).
- Indicadores técnicos + indicadores de contexto (força relativa, beta e correlação com o mercado).
- Otimização de hiperparâmetros (RandomizedSearchCV com validação temporal).
- Avaliação walk-forward fora da amostra, comparação com baselines e backtest.
- Previsão para vários horizontes com intervalo de confiança e probabilidade de alta.

AVISO: projeto educacional. Nada aqui é recomendação de investimento.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

DIAS_UTEIS_ANO = 252
Progresso = Callable[[float, str], None]


def _sem_progresso(_frac: float, _msg: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# 1. Dados da ação
# --------------------------------------------------------------------------- #
def normalizar_ticker(ticker: str) -> str:
    """'petr4' -> 'PETR4.SA' (padrão de ações da B3: 4 letras + 1 ou 2 dígitos)."""
    t = ticker.strip().upper()
    if re.fullmatch(r"[A-Z]{4}\d{1,2}", t):
        t += ".SA"
    return t


def eh_brasileira(ticker: str) -> bool:
    return ticker.upper().endswith(".SA")


def baixar_yahoo(ticker: str, anos: int) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=f"{anos}y", auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Não foi possível baixar dados de '{ticker}' do Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def ler_csv(arquivo) -> pd.DataFrame:
    """CSV com colunas Date/Data, Open, High, Low, Close, Volume (maiúsculas/minúsculas tanto faz)."""
    df = pd.read_csv(arquivo)
    df.columns = [c.strip().capitalize() for c in df.columns]
    data_col = "Date" if "Date" in df.columns else "Data"
    df[data_col] = pd.to_datetime(df[data_col])
    df = df.set_index(data_col).sort_index()
    for col in ("Open", "High", "Low"):
        if col not in df.columns:
            df[col] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).dropna()


# --------------------------------------------------------------------------- #
# 2. Dados externos
# --------------------------------------------------------------------------- #
@dataclass
class SerieExterna:
    nome: str        # nome curto usado nas features
    descricao: str
    yahoo: str | None = None
    tipo: str = "preco"   # "preco" -> usa retornos; "nivel" -> usa nível e variação (juros, VIX)
    mercado: bool = False  # índice de referência para beta / força relativa
    limite_ffill: int = 10  # pregões que um valor pode ser repetido (séries mensais precisam de mais)


def series_externas(ticker: str) -> list[SerieExterna]:
    comuns = [
        SerieExterna("sp500", "S&P 500", "^GSPC", mercado=not eh_brasileira(ticker)),
        SerieExterna("vix", "VIX (medo do mercado)", "^VIX", tipo="nivel"),
        SerieExterna("petroleo", "Petróleo Brent", "BZ=F"),
    ]
    if eh_brasileira(ticker):
        return [
            SerieExterna("ibov", "Ibovespa", "^BVSP", mercado=True),
            SerieExterna("dolar", "Dólar (USD/BRL)", "BRL=X"),
            SerieExterna("selic", "Taxa Selic (Banco Central)", None, tipo="nivel"),
            SerieExterna("focus_ipca", "Expectativa de inflação 12m (Focus/BC)", None, tipo="nivel"),
            SerieExterna("incerteza", "Incerteza política e econômica (EPU Brasil)", None,
                         tipo="nivel", limite_ffill=45),
            *comuns,
        ]
    return [
        *comuns,
        SerieExterna("nasdaq", "Nasdaq", "^IXIC"),
        SerieExterna("juros_eua", "Treasury 10 anos (EUA)", "^TNX", tipo="nivel"),
        SerieExterna("dxy", "Índice do dólar (DXY)", "DX-Y.NYB"),
    ]


PASTA_CACHE = Path(__file__).with_name("cache_dados")
NAVEGADOR = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def http_get(url: str, params: dict | None = None, timeout=(15, 60)):
    """GET com cabeçalho de navegador e novas tentativas (os servidores do BC são instáveis)."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    sessao = requests.Session()
    sessao.headers.update(NAVEGADOR)
    tentativas = Retry(total=3, connect=3, read=3, backoff_factor=2,
                       status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    sessao.mount("https://", HTTPAdapter(max_retries=tentativas))
    if params:
        # Codifica espaços como %20 (e não "+"): a API OData do Banco Central rejeita "+" com erro 400
        from urllib.parse import quote, urlencode
        url = f"{url}?{urlencode(params, quote_via=quote, safe="$',()")}"
    r = sessao.get(url, timeout=timeout)
    r.raise_for_status()
    return r


def erro_curto(e: Exception) -> str:
    """Mensagem de erro legível para o usuário."""
    import requests

    if isinstance(e, requests.exceptions.Timeout):
        return "o servidor demorou demais para responder"
    if isinstance(e, requests.exceptions.ConnectionError):
        return "não foi possível conectar ao servidor"
    if isinstance(e, requests.exceptions.HTTPError):
        return f"o servidor respondeu com erro {e.response.status_code}"
    return str(e)[:150]


def _salvar_cache(nome: str, serie: pd.Series):
    try:
        PASTA_CACHE.mkdir(exist_ok=True)
        serie.rename("valor").to_csv(PASTA_CACHE / f"{nome}.csv", index_label="data")
    except OSError:
        pass


def _ler_cache(nome: str) -> pd.Series | None:
    arq = PASTA_CACHE / f"{nome}.csv"
    if not arq.exists():
        return None
    tab = pd.read_csv(arq, parse_dates=["data"])
    return pd.Series(tab["valor"].values, index=tab["data"])


def baixar_selic(inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.Series:
    """Meta Selic (SGS 432) do Banco Central. A API limita consultas diárias a 10 anos, então baixa em blocos."""
    partes = []
    ini = inicio
    while ini <= fim:
        fim_bloco = min(ini + pd.DateOffset(years=5), fim)
        r = http_get("https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados",
                      {"formato": "json", "dataInicial": f"{ini:%d/%m/%Y}", "dataFinal": f"{fim_bloco:%d/%m/%Y}"})
        partes.append(pd.DataFrame(r.json()))
        ini = fim_bloco + pd.Timedelta(days=1)
    dados = pd.concat(partes)
    s = pd.Series(
        dados["valor"].astype(float).values,
        index=pd.to_datetime(dados["data"], format="%d/%m/%Y"),
    )
    return s[~s.index.duplicated()].sort_index()


def baixar_focus_ipca(inicio: pd.Timestamp) -> pd.Series:
    """Mediana da expectativa de IPCA 12 meses à frente do boletim Focus (Banco Central)."""
    url = ("https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
           "ExpectativasMercadoInflacao12Meses")
    params = {
        "$filter": f"Indicador eq 'IPCA' and Suavizada eq 'S' and baseCalculo eq 0 "
                   f"and Data ge '{inicio:%Y-%m-%d}'",
        "$select": "Data,Mediana", "$orderby": "Data", "$top": "20000", "$format": "json",
    }
    dados = pd.DataFrame(http_get(url, params).json()["value"])
    s = pd.Series(dados["Mediana"].astype(float).values, index=pd.to_datetime(dados["Data"]))
    return s[~s.index.duplicated(keep="last")].sort_index()


def baixar_incerteza_politica() -> pd.Series:
    """
    Índice de Incerteza de Política Econômica do Brasil (Baker, Bloom & Davis — policyuncertainty.com),
    mensal, baseado em notícias. Cada mês só é usado a partir do mês seguinte (quando é publicado).
    """
    import io

    r = http_get("https://www.policyuncertainty.com/media/Brazil_Policy_Uncertainty_Data.xlsx")
    return ler_planilha_epu(io.BytesIO(r.content))


def ler_planilha_epu(arquivo) -> pd.Series:
    """
    Lê a planilha do EPU sem depender do nome exato das colunas: procura a linha de cabeçalho
    com 'year'/'month' (ou uma coluna de data) e usa a primeira coluna numérica restante como índice.
    """
    bruto = pd.read_excel(arquivo, header=None)
    norm = lambda v: str(v).strip().lower()
    linha_cab = next((i for i in range(min(len(bruto), 15))
                      if any(norm(v) in ("year", "ano") for v in bruto.iloc[i])), None)
    if linha_cab is not None:
        tab = bruto.iloc[linha_cab + 1:].copy()
        tab.columns = [norm(v) for v in bruto.iloc[linha_cab]]
        col_ano = next(c for c in tab.columns if c in ("year", "ano"))
        col_mes = next(c for c in tab.columns if c in ("month", "mes", "mês"))
        ano = pd.to_numeric(tab[col_ano], errors="coerce")
        mes = pd.to_numeric(tab[col_mes], errors="coerce")
        outras = [c for c in tab.columns if c not in (col_ano, col_mes)]
        ok = ano.notna() & mes.notna()
        datas = pd.to_datetime(pd.DataFrame({"year": ano[ok], "month": mes[ok], "day": 1}).astype(int))
        tab = tab[ok]
    else:
        # Formato alternativo: primeira coluna com datas
        tab = bruto.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            datas_brutas = pd.to_datetime(tab.iloc[:, 0], errors="coerce")
        ok = datas_brutas.notna()
        tab, datas = tab[ok], datas_brutas[ok].dt.to_period("M").dt.to_timestamp()
        outras = list(tab.columns[1:])
    col_valor = next((c for c in outras if pd.to_numeric(tab[c], errors="coerce").notna().mean() > 0.8), None)
    if col_valor is None:
        raise RuntimeError(f"formato da planilha não reconhecido (colunas: {list(bruto.iloc[0])[:5]})")
    valor = pd.to_numeric(tab[col_valor], errors="coerce").values
    s = pd.Series(valor, index=pd.DatetimeIndex(datas) + pd.DateOffset(months=1)).dropna()
    return s[~s.index.duplicated()].sort_index()


def baixar_externos(ticker: str, inicio: pd.Timestamp, fim: pd.Timestamp):
    """Retorna (DataFrame com uma coluna por série, lista de séries obtidas, lista de avisos)."""
    import yfinance as yf

    series = series_externas(ticker)
    dados, avisos = {}, []
    codigos = [s.yahoo for s in series if s.yahoo]
    try:
        bruto = yf.download(
            codigos, start=inicio - pd.Timedelta(days=400), end=fim + pd.Timedelta(days=1),
            auto_adjust=True, progress=False,
        )
        close = bruto["Close"] if isinstance(bruto.columns, pd.MultiIndex) else bruto[["Close"]]
        close.index = pd.to_datetime(close.index).tz_localize(None)
    except Exception as e:
        close = pd.DataFrame()
        avisos.append(f"Falha ao baixar dados externos do Yahoo: {e}")

    for s in series:
        try:
            if s.yahoo:
                serie = close[s.yahoo].dropna() if s.yahoo in close.columns else pd.Series(dtype=float)
            else:
                baixar = {
                    "selic": lambda: baixar_selic(inicio - pd.Timedelta(days=400), fim),
                    "focus_ipca": lambda: baixar_focus_ipca(inicio - pd.Timedelta(days=400)),
                    "incerteza": baixar_incerteza_politica,
                }[s.nome]
                try:
                    serie = baixar()
                    _salvar_cache(s.nome, serie)
                except Exception as e:
                    serie = _ler_cache(s.nome)
                    if serie is None:
                        raise
                    avisos.append(f"{s.descricao}: fonte fora do ar ({type(e).__name__}); "
                                  f"usando dados salvos até {serie.index.max():%d/%m/%Y}.")
            if len(serie) < (24 if s.limite_ffill > 10 else 100):
                raise RuntimeError("dados insuficientes")
            dados[s.nome] = serie
        except Exception as e:
            avisos.append(f"{s.descricao} indisponível ({erro_curto(e)}).")

    obtidas = [s for s in series if s.nome in dados]
    return pd.DataFrame(dados), obtidas, avisos


# --------------------------------------------------------------------------- #
# 3. Dados sintéticos (demonstração)
# --------------------------------------------------------------------------- #
def gerar_demo(anos: int, seed: int = 7):
    """
    Ação + mercado + dólar + juros sintéticos. A ação tem beta com o mercado e um efeito
    defasado (pequeno) do mercado e do dólar — ou seja, há sinal real a ser descoberto.
    Serve para validar que o pipeline encontra padrões quando eles existem.
    """
    rng = np.random.default_rng(seed)
    n = anos * DIAS_UTEIS_ANO
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)

    vol_regime = 0.010 + 0.008 * (np.sin(np.arange(n) / 150) > 0.4)
    r_mkt = 0.0003 + vol_regime * rng.standard_normal(n)
    r_usd = 0.0001 + 0.009 * rng.standard_normal(n) - 0.25 * r_mkt
    r_sp = 0.0003 + 0.009 * rng.standard_normal(n) + 0.3 * r_mkt
    vix = 18 + 400 * pd.Series(vol_regime).rolling(5, min_periods=1).mean().values + rng.normal(0, 1.5, n)
    selic = np.round(np.clip(10 + np.cumsum(rng.choice([0, 0, 0, 0.25, -0.25], n, p=[.96, .01, .01, .01, .01])), 2, 15), 2)

    focus_ipca = 4 + np.cumsum(rng.normal(0, 0.02, n))
    # Incerteza política mensal (constante dentro do mês), com efeito negativo leve nos retornos
    meses = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n).to_period("M")
    epu_mes = pd.Series(np.exp(rng.normal(5, 0.35, meses.nunique())), index=meses.unique())
    incerteza = epu_mes.reindex(meses).values
    z_inc = (incerteza - incerteza.mean()) / incerteza.std()

    r_mkt_lag = np.r_[0, r_mkt[:-1]]
    r_usd_lag = np.r_[0, r_usd[:-1]]
    r_acao = (0.0002 + 1.1 * r_mkt + 0.15 * r_mkt_lag - 0.12 * r_usd_lag - 0.0006 * z_inc
              + 0.012 * rng.standard_normal(n))

    close = 25 * np.exp(np.cumsum(r_acao))
    abertura = close * np.exp(rng.normal(0, 0.004, n))
    acao = pd.DataFrame({
        "Open": abertura,
        "High": np.maximum(abertura, close) * np.exp(np.abs(rng.normal(0, 0.006, n))),
        "Low": np.minimum(abertura, close) * np.exp(-np.abs(rng.normal(0, 0.006, n))),
        "Close": close,
        "Volume": rng.lognormal(15, 0.4, n) * (1 + 20 * np.abs(r_acao)),
    }, index=idx)

    externos = pd.DataFrame({
        "ibov": 100_000 * np.exp(np.cumsum(r_mkt)),
        "dolar": 5 * np.exp(np.cumsum(r_usd)),
        "sp500": 4000 * np.exp(np.cumsum(r_sp)),
        "vix": vix,
        "selic": selic,
        "focus_ipca": focus_ipca,
        "incerteza": incerteza,
    }, index=idx)
    obtidas = [s for s in series_externas("DEMO.SA") if s.nome in externos.columns]
    return acao, externos, obtidas


# --------------------------------------------------------------------------- #
# 4. Features
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, periodo: int = 14) -> pd.Series:
    delta = close.diff()
    ganho = delta.clip(lower=0).ewm(alpha=1 / periodo, adjust=False).mean()
    perda = (-delta.clip(upper=0)).ewm(alpha=1 / periodo, adjust=False).mean()
    return 100 - 100 / (1 + ganho / perda.replace(0, np.nan))


def features_tecnicas(df: pd.DataFrame) -> pd.DataFrame:
    c = df["Close"]
    log_ret = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    for lag in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret_{lag}d"] = np.log(c / c.shift(lag))
    for janela in (5, 10, 20, 50, 200):
        f[f"dist_sma_{janela}"] = c / c.rolling(janela).mean() - 1

    f["vol_10d"] = log_ret.rolling(10).std()
    f["vol_20d"] = log_ret.rolling(20).std()
    f["vol_60d"] = log_ret.rolling(60).std()
    f["vol_ratio"] = f["vol_10d"] / f["vol_60d"]
    f["rsi_14"] = rsi(c, 14) / 100

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd"] = macd / c
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c

    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bollinger_pct_b"] = (c - (sma20 - 2 * std20)) / (4 * std20)
    f["dist_max_52s"] = c / c.rolling(252, min_periods=60).max() - 1
    f["dist_min_52s"] = c / c.rolling(252, min_periods=60).min() - 1

    f["amplitude_dia"] = (df["High"] - df["Low"]) / c
    f["gap_abertura"] = np.log(df["Open"] / c.shift(1))
    if df["Volume"].sum() > 0:
        vol = df["Volume"].replace(0, np.nan)
        f["volume_rel_20d"] = np.log(vol / vol.rolling(20).mean())

    f["dia_semana"] = df.index.dayofweek
    f["mes"] = df.index.month
    return f


def alinhar(externos: pd.DataFrame, indice: pd.DatetimeIndex,
            series: list[SerieExterna] | None = None) -> pd.DataFrame:
    """Leva as séries externas para as datas de pregão da ação, usando o último valor conhecido."""
    limites = {s.nome: s.limite_ffill for s in series or []}
    uniao = externos.index.union(indice)
    base = externos.reindex(uniao)
    return pd.DataFrame(
        {c: base[c].ffill(limit=limites.get(c, 10)) for c in base.columns}, index=uniao
    ).reindex(indice)


def features_externas(df: pd.DataFrame, externos: pd.DataFrame, series: list[SerieExterna]) -> pd.DataFrame:
    ext = alinhar(externos, df.index, series)
    f = pd.DataFrame(index=df.index)
    ret_acao = np.log(df["Close"]).diff()

    for s in series:
        x = ext[s.nome]
        if s.tipo == "preco":
            for lag in (1, 5, 20):
                f[f"{s.nome}_ret_{lag}d"] = np.log(x / x.shift(lag))
            f[f"{s.nome}_dist_sma_50"] = x / x.rolling(50).mean() - 1
        else:
            f[f"{s.nome}_nivel"] = x
            f[f"{s.nome}_var_5d"] = x - x.shift(5)
            f[f"{s.nome}_var_20d"] = x - x.shift(20)

        if s.mercado:
            ret_mkt = np.log(x).diff()
            f["forca_rel_20d"] = np.log(df["Close"] / df["Close"].shift(20)) - np.log(x / x.shift(20))
            f["forca_rel_60d"] = np.log(df["Close"] / df["Close"].shift(60)) - np.log(x / x.shift(60))
            f["beta_60d"] = ret_acao.rolling(60).cov(ret_mkt) / ret_mkt.rolling(60).var()
            f["correl_60d"] = ret_acao.rolling(60).corr(ret_mkt)
    return f


def montar_dataset(df, externos, series, horizonte):
    """
    X no dia t (só informação disponível até o fechamento de t);
    alvo = log(Close[t+h] / Close[t]). Também devolve a última linha para a previsão.
    """
    X = features_tecnicas(df)
    if externos is not None and len(series) > 0:
        X = X.join(features_externas(df, externos, series))
    X = X.replace([np.inf, -np.inf], np.nan)
    # Remove colunas com muitas falhas (ex.: série externa com histórico curto)
    X = X.loc[:, X.iloc[260:].isna().mean() < 0.2]
    y = np.log(df["Close"].shift(-horizonte) / df["Close"])
    validas = X.notna().all(axis=1)
    X, y = X[validas], y[validas]
    return X[y.notna()], y[y.notna()], X.iloc[[-1]]


# --------------------------------------------------------------------------- #
# 5. Modelos e otimização
# --------------------------------------------------------------------------- #
def modelos_base() -> dict:
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=20, max_features=0.5,
            n_jobs=-1, random_state=0,
        ),
        "GradientBoosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_depth=3, min_samples_leaf=40,
            l2_regularization=1.0, random_state=0,
        ),
    }


ESPACOS_BUSCA = {
    "Ridge": {"ridge__alpha": loguniform(1e-2, 1e4)},
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 6, 8, 12],
        "min_samples_leaf": [10, 20, 50, 100, 200],
        "max_features": [0.2, 0.33, 0.5, 0.8],
    },
    "GradientBoosting": {
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "max_iter": [100, 200, 400],
        "max_depth": [2, 3, 4, 6],
        "min_samples_leaf": [20, 50, 100, 200],
        "l2_regularization": [0.0, 0.5, 1.0, 5.0, 20.0],
    },
}


def aparar(y: pd.Series, q: float = 0.01) -> pd.Series:
    """
    Limita o alvo de TREINO aos percentis 1%-99%. Dias extremos (crashes, notícias pontuais)
    dominariam o erro quadrático e fariam o modelo perseguir ruído. A avaliação usa o alvo real.
    """
    lo, hi = y.quantile([q, 1 - q])
    return y.clip(lo, hi)


def otimizar(X, y, horizonte, n_iter, progresso: Progresso, base=0.0, peso=1.0) -> tuple[dict, dict]:
    """Busca aleatória de hiperparâmetros com validação temporal (com intervalo `gap` = horizonte)."""
    modelos, melhores = {}, {}
    cv = TimeSeriesSplit(n_splits=4, gap=horizonte)
    nomes = list(modelos_base())
    for i, (nome, modelo) in enumerate(modelos_base().items()):
        progresso(base + peso * i / len(nomes), f"Otimizando {nome}...")
        if nome == "RandomForest":
            modelo.set_params(n_jobs=1)  # o paralelismo fica com a busca
        busca = RandomizedSearchCV(
            modelo, ESPACOS_BUSCA[nome], n_iter=n_iter, cv=cv,
            scoring="neg_mean_squared_error", random_state=0, n_jobs=-1,
        )
        busca.fit(X, aparar(y))
        melhor = busca.best_estimator_
        if nome == "RandomForest":
            melhor.set_params(n_jobs=-1)
        modelos[nome] = melhor
        melhores[nome] = {k.replace("ridge__", ""): (float(round(v, 4)) if isinstance(v, float) else v)
                          for k, v in busca.best_params_.items()}
    return modelos, melhores


class Ensemble:
    """Média das previsões dos modelos individuais."""

    def __init__(self, modelos: dict):
        self.modelos = modelos

    def fit(self, X, y):
        for m in self.modelos.values():
            m.fit(X, y)
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.modelos.values()], axis=0)


def instanciar(modelos: dict) -> dict:
    novos = {nome: clone(m) for nome, m in modelos.items()}
    novos["Ensemble"] = Ensemble({nome: clone(m) for nome, m in modelos.items()})
    return novos


# --------------------------------------------------------------------------- #
# 6. Avaliação walk-forward
# --------------------------------------------------------------------------- #
def walk_forward(X, y, modelos, horizonte, inicio_teste, n_blocos, progresso, base, peso):
    """Treina só com o passado (até `horizonte` dias antes do bloco) e prevê o bloco seguinte."""
    limites = np.linspace(inicio_teste, len(X), n_blocos + 1).astype(int)
    blocos = []
    for b in range(n_blocos):
        progresso(base + peso * b / n_blocos, f"Validação walk-forward: bloco {b + 1}/{n_blocos}")
        ini, fim = limites[b], limites[b + 1]
        treino = slice(0, ini - horizonte)
        bloco = pd.DataFrame({"real": y.iloc[ini:fim]})
        for nome, m in instanciar(modelos).items():
            m.fit(X.iloc[treino], aparar(y.iloc[treino]))
            bloco[nome] = m.predict(X.iloc[ini:fim])
        bloco["Baseline: retorno zero"] = 0.0
        bloco["Baseline: tendência 20d"] = X["ret_20d"].iloc[ini:fim].values / 20 * horizonte
        blocos.append(bloco)
    return pd.concat(blocos)


def calcular_metricas(prev: pd.DataFrame, horizonte: int):
    real = prev["real"]
    passo = prev.iloc[::horizonte]  # operações sem sobreposição no backtest
    anual = DIAS_UTEIS_ANO / horizonte
    linhas, curvas = [], {}
    for nome in prev.columns.drop("real"):
        p = prev[nome]
        acerto = np.mean(np.sign(p) == np.sign(real)) * 100 if p.abs().sum() > 0 else np.nan
        ret_estr = pd.Series(np.where(passo[nome] > 0, passo["real"], 0.0), index=passo.index)
        sharpe = ret_estr.mean() / ret_estr.std() * np.sqrt(anual) if ret_estr.std() > 0 else np.nan
        curvas[nome] = np.exp(ret_estr.cumsum())
        linhas.append({
            "Modelo": nome,
            "MAE (%)": mean_absolute_error(real, p) * 100,
            "RMSE (%)": np.sqrt(mean_squared_error(real, p)) * 100,
            "Acerto de direção (%)": acerto,
            "Correlação": np.corrcoef(p, real)[0, 1] if p.std() > 0 else np.nan,
            "Retorno da estratégia (%)": (np.exp(ret_estr.sum()) - 1) * 100,
            "Sharpe": sharpe,
        })
    curvas["Comprar e segurar"] = np.exp(passo["real"].cumsum())
    tabela = pd.DataFrame(linhas).set_index("Modelo")
    tabela.loc["Comprar e segurar", "Retorno da estratégia (%)"] = (np.exp(passo["real"].sum()) - 1) * 100
    bh = passo["real"]
    tabela.loc["Comprar e segurar", "Sharpe"] = bh.mean() / bh.std() * np.sqrt(anual)
    return tabela, pd.DataFrame(curvas)


# --------------------------------------------------------------------------- #
# 7. Orquestração
# --------------------------------------------------------------------------- #
@dataclass
class ResultadoHorizonte:
    horizonte: int
    data_alvo: pd.Timestamp
    preco_atual: float
    preco_previsto: float
    retorno_previsto: float          # em %
    intervalo_80: tuple[float, float]
    intervalo_95: tuple[float, float]
    prob_alta: float                 # 0..1
    melhor_modelo: str
    ganho_vs_baseline: float         # % de redução do RMSE vs. retorno zero (>0 = melhor)
    acerto_direcao: float
    confiabilidade: str
    metricas: pd.DataFrame
    curvas: pd.DataFrame
    previsoes_oos: pd.DataFrame
    importancias: pd.Series
    parametros: dict
    n_features: int
    n_amostras: int
    sinal_forte: bool = False        # previsão atual está entre as 25% mais intensas do teste
    acerto_sinal_forte: float = float("nan")   # acerto de direção no teste quando o sinal foi forte
    n_sinal_forte: int = 0
    p_valor_sinal_forte: float = 1.0  # chance de um acerto desses ser pura sorte (moeda)


def analisar_sinais(prev: pd.DataFrame, modelo: str, ret_atual: float, quantil: float = 0.75):
    """
    Mede o acerto só nas previsões mais intensas do teste (|retorno previsto| no quartil superior).
    Quando o modelo prevê movimentos maiores, costuma acertar a direção com mais frequência.
    O teste binomial diz se esse acerto é melhor que jogar uma moeda.
    """
    from scipy.stats import binomtest

    p, real = prev[modelo], prev["real"]
    if p.abs().sum() == 0:
        return False, float("nan"), 0, 1.0
    limiar = p.abs().quantile(quantil)
    forte = p.abs() >= limiar
    acertos = int((np.sign(p[forte]) == np.sign(real[forte])).sum())
    n = int(forte.sum())
    p_valor = binomtest(acertos, n, 0.5, alternative="greater").pvalue if n else 1.0
    return bool(abs(ret_atual) >= limiar), acertos / n * 100 if n else float("nan"), n, float(p_valor)


def classificar_confiabilidade(ganho: float, acerto: float, correl: float) -> str:
    if ganho > 2 and acerto > 54 and correl > 0.1:
        return "Moderada"
    if ganho > 0 and acerto > 51:
        return "Baixa"
    return "Muito baixa"


def prever_horizonte(df, externos, series, horizonte, otimizar_params=True, n_iter=10,
                     n_blocos=5, frac_teste=0.4, progresso: Progresso = _sem_progresso,
                     base=0.0, peso=1.0) -> ResultadoHorizonte:
    X, y, X_ultimo = montar_dataset(df, externos, series, horizonte)
    if len(X) < 400:
        raise ValueError("Histórico insuficiente: são necessários ~400 pregões após o cálculo dos indicadores.")

    inicio_teste = int(len(X) * (1 - frac_teste))

    # Hiperparâmetros escolhidos SÓ com dados anteriores ao período de teste (sem vazamento)
    if otimizar_params:
        modelos, params = otimizar(X.iloc[:inicio_teste - horizonte], y.iloc[:inicio_teste - horizonte],
                                   horizonte, n_iter, progresso, base, peso * 0.5)
    else:
        modelos = modelos_base()
        params = {n: {k.split("__")[-1]: v for k, v in m.get_params().items()
                      if k.split("__")[-1] in {"alpha", "n_estimators", "max_depth", "min_samples_leaf",
                                                "max_features", "learning_rate", "max_iter",
                                                "l2_regularization"}}
                  for n, m in modelos.items()}

    b2 = base + peso * (0.5 if otimizar_params else 0.0)
    p2 = peso * (0.4 if otimizar_params else 0.85)
    prev = walk_forward(X, y, modelos, horizonte, inicio_teste, n_blocos, progresso, b2, p2)
    tabela, curvas = calcular_metricas(prev, horizonte)

    candidatos = tabela.loc[["Ridge", "RandomForest", "GradientBoosting", "Ensemble"]]
    melhor = candidatos["RMSE (%)"].idxmin()
    rmse_zero = tabela.loc["Baseline: retorno zero", "RMSE (%)"]
    ganho = (1 - candidatos.loc[melhor, "RMSE (%)"] / rmse_zero) * 100

    progresso(base + peso * 0.92, "Treinando modelo final com todo o histórico...")
    finais = instanciar(modelos)
    final = finais[melhor]
    final.fit(X, aparar(y))
    ret = float(final.predict(X_ultimo)[0])

    # Importância dos indicadores: Random Forest final (mesmos dados)
    if melhor == "RandomForest":
        rf = final
    elif melhor == "Ensemble":
        rf = final.modelos["RandomForest"]
    else:
        rf = finais["RandomForest"].fit(X, aparar(y))
    importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Incerteza: distribuição empírica dos erros fora da amostra
    erros = (prev["real"] - prev[melhor]).values
    cenarios = ret + erros
    preco_atual = float(df["Close"].iloc[-1])
    q = np.quantile(cenarios, [0.025, 0.10, 0.90, 0.975])
    acerto = tabela.loc[melhor, "Acerto de direção (%)"]
    sinal_forte, acerto_forte, n_forte, p_forte = analisar_sinais(prev, melhor, ret)

    return ResultadoHorizonte(
        horizonte=horizonte,
        data_alvo=df.index[-1] + pd.offsets.BDay(horizonte),
        preco_atual=preco_atual,
        preco_previsto=preco_atual * np.exp(ret),
        retorno_previsto=(np.exp(ret) - 1) * 100,
        intervalo_80=(preco_atual * np.exp(q[1]), preco_atual * np.exp(q[2])),
        intervalo_95=(preco_atual * np.exp(q[0]), preco_atual * np.exp(q[3])),
        prob_alta=float(np.mean(cenarios > 0)),
        melhor_modelo=melhor,
        ganho_vs_baseline=ganho,
        acerto_direcao=acerto,
        confiabilidade=classificar_confiabilidade(ganho, acerto, tabela.loc[melhor, "Correlação"]),
        metricas=tabela,
        curvas=curvas,
        previsoes_oos=prev,
        importancias=importancias,
        parametros=params,
        n_features=X.shape[1],
        n_amostras=len(X),
        sinal_forte=sinal_forte,
        acerto_sinal_forte=acerto_forte,
        n_sinal_forte=n_forte,
        p_valor_sinal_forte=p_forte,
    )


def prever(df, externos, series, horizontes, otimizar_params=True, n_iter=10, n_blocos=5,
           progresso: Progresso = _sem_progresso) -> dict[int, ResultadoHorizonte]:
    resultados = {}
    for i, h in enumerate(horizontes):
        def prog(frac, msg, h=h):
            progresso(frac, f"[{h} pregão(ões)] {msg}")
        resultados[h] = prever_horizonte(
            df, externos, series, h, otimizar_params, n_iter, n_blocos,
            progresso=prog, base=i / len(horizontes), peso=1 / len(horizontes),
        )
    progresso(1.0, "Concluído")
    return resultados
