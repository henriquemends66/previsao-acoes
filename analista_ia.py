"""
Analista de IA (Claude) para o contexto qualitativo da ação.

Etapa 1 — pesquisa: o Claude usa a busca na web para levantar notícias recentes sobre a empresa,
o setor, o cenário político, a macroeconomia e os rumores/especulações do mercado.
Etapa 2 — síntese: transforma a pesquisa + os resultados do modelo quantitativo em uma análise
estruturada (fatores, catalisadores e cenários otimista/base/pessimista com probabilidades).

Requer uma chave da API da Anthropic (variável ANTHROPIC_API_KEY ou informada no app).
A análise é qualitativa e NÃO foi validada historicamente: use o acompanhamento de previsões
do app para medir se ela acerta mais do que o modelo quantitativo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import anthropic

MODELOS = {
    "claude-opus-5-5": "Claude Opus 5.5 (mais preciso)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 (mais barato)",
}
# US$ por milhão de tokens (entrada, saída) e por busca na web
PRECOS = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0)}
PRECO_BUSCA = 10.0 / 1000
BETA_FALLBACK = "server-side-fallback-2026-07-01"


class ErroAnalise(Exception):
    pass


@dataclass
class AnaliseIA:
    dados: dict
    pesquisa: str
    fontes: list[dict]
    modelo: str
    custo_usd: float
    buscas: int
    preco_atual: float
    preco_esperado: float = field(init=False)
    retorno_esperado: float = field(init=False)

    def __post_init__(self):
        cen = self.dados["cenarios"]
        total = sum(c["probabilidade"] for c in cen) or 1
        self.preco_esperado = sum(c["probabilidade"] * c["preco_alvo"] for c in cen) / total
        self.retorno_esperado = (self.preco_esperado / self.preco_atual - 1) * 100


# --------------------------------------------------------------------------- #
# Esquema da resposta estruturada
# --------------------------------------------------------------------------- #
def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _enum(*valores) -> dict:
    return {"type": "string", "enum": list(valores)}


ESQUEMA = _obj({
    "resumo": {"type": "string", "description": "3 a 5 frases com a leitura geral do momento da ação."},
    "sentimento_geral": _enum("muito_negativo", "negativo", "neutro", "positivo", "muito_positivo"),
    "fatores": {"type": "array", "items": _obj({
        "categoria": _enum("politico", "macroeconomico", "setorial", "empresa", "especulacao"),
        "descricao": {"type": "string"},
        "impacto": _enum("alta", "baixa", "neutro"),
        "intensidade": _enum("fraca", "moderada", "forte"),
        "confirmado": {"type": "boolean", "description": "false para rumores e especulações não confirmados."},
        "fonte": {"type": "string", "description": "Veículo e data da notícia."},
    })},
    "catalisadores": {"type": "array", "items": _obj({
        "evento": {"type": "string"},
        "data": {"type": "string", "description": "Data ou período previsto (ex.: 05/11/2026, 'início de dezembro')."},
        "por_que_importa": {"type": "string"},
    })},
    "cenarios": {"type": "array", "items": _obj({
        "nome": _enum("otimista", "base", "pessimista"),
        "probabilidade": {"type": "number", "description": "Em %, os três somam 100."},
        "preco_alvo": {"type": "number"},
        "premissas": {"type": "string"},
    })},
    "concordancia_com_modelo": _enum("concorda", "discorda_parcialmente", "discorda"),
    "comentario_modelo": {"type": "string",
                          "description": "Como o contexto atual reforça ou contradiz a previsão quantitativa."},
    "riscos_principais": {"type": "array", "items": {"type": "string"}},
    "confianca": _enum("baixa", "media", "alta"),
})


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
SISTEMA_PESQUISA = """Você é um analista de ações experiente que cobre o mercado brasileiro e global.
Sua tarefa agora é PESQUISAR, não recomendar. Use a busca na web para levantar o que aconteceu
nas últimas duas semanas e o que está previsto para as próximas semanas que possa mover a ação.

Cubra, nesta ordem:
1. Empresa: resultados, projeções, dividendos, fusões/aquisições, troca de gestão, processos, investigações.
2. Setor e commodities relevantes.
3. Cenário político: governo, Congresso, STF, eleições, interferência em estatais, regulação, impostos, política fiscal.
4. Macroeconomia: juros (Copom/Fed), inflação, câmbio, fluxo de investidor estrangeiro, risco-país.
5. Especulações e rumores do mercado: o que se comenta sem confirmação oficial, posicionamento de grandes investidores,
   recomendações recentes de bancos e corretoras.
6. Agenda: datas de balanço, reuniões do Copom/Fed, votações, eventos regulatórios.

Escreva notas objetivas em português, com data e veículo de cada informação. Separe claramente fatos
confirmados de rumores. Não invente informações: se não encontrar algo, diga que não encontrou."""

SISTEMA_SINTESE = """Você é um analista de ações. Transforme a pesquisa e os dados quantitativos recebidos
em uma análise estruturada, em português do Brasil.

Regras:
- Use apenas informações presentes na pesquisa e nos dados fornecidos.
- Marque rumores e especulações com confirmado=false.
- Os três cenários (otimista, base, pessimista) devem ter probabilidades que somam 100 e preços-alvo
  coerentes com a volatilidade histórica informada para o horizonte pedido. Evite cenários extremos sem um motivo concreto.
- O modelo quantitativo costuma ter poder de previsão baixo; não o trate como certeza, mas também não o ignore.
- Seja calibrado: se não houver informação forte, a confiança deve ser baixa e o cenário base próximo do preço atual."""


def _texto(resposta) -> str:
    return "\n".join(b.text for b in resposta.content if b.type == "text").strip()


def _fontes(conteudo) -> list[dict]:
    fontes, vistos = [], set()
    for b in conteudo:
        if b.type == "web_search_tool_result" and isinstance(b.content, list):
            for r in b.content:
                if getattr(r, "url", None) and r.url not in vistos:
                    vistos.add(r.url)
                    fontes.append({"titulo": r.title, "url": r.url, "data": getattr(r, "page_age", None)})
    return fontes


def _custo(modelo: str, usos: list) -> tuple[float, int]:
    p_in, p_out = PRECOS[modelo]
    custo, buscas = 0.0, 0
    for u in usos:
        entrada = (u.input_tokens + (u.cache_creation_input_tokens or 0) * 1.25
                   + (u.cache_read_input_tokens or 0) * 0.1)
        custo += entrada * p_in / 1e6 + u.output_tokens * p_out / 1e6
        stu = getattr(u, "server_tool_use", None)
        buscas += (getattr(stu, "web_search_requests", 0) or 0) if stu else 0
    return custo + buscas * PRECO_BUSCA, buscas


def _checar_recusa(resposta):
    if resposta.stop_reason == "refusal":
        detalhe = getattr(resposta, "stop_details", None)
        raise ErroAnalise(f"O modelo recusou a solicitação ({getattr(detalhe, 'explanation', '') or 'sem detalhes'}).")


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #
def analisar(contexto_texto: str, ticker: str, preco_atual: float, horizonte_desc: str,
             chave_api: str | None = None, modelo: str = "claude-opus-5-5",
             max_buscas: int = 8, progresso=lambda msg: None) -> AnaliseIA:
    client = anthropic.Anthropic(api_key=chave_api) if chave_api else anthropic.Anthropic()
    usos = []
    try:
        # Etapa 1: pesquisa com busca na web
        progresso("Pesquisando notícias, cenário político e especulações na web...")
        mensagens = [{"role": "user", "content": (
            f"Ação: {ticker}\nPreço atual: {preco_atual:.2f}\n\n"
            f"Dados que já temos (use como ponto de partida e aprofunde):\n{contexto_texto}"
        )}]
        conteudo_total = []
        for _ in range(4):  # retoma turnos pausados pela busca (pause_turn)
            with client.beta.messages.stream(
                model=modelo, max_tokens=32000, system=SISTEMA_PESQUISA, messages=mensagens,
                tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": max_buscas}],
                output_config={"effort": "medium"},
                betas=[BETA_FALLBACK], fallbacks="default",
            ) as stream:
                resp = stream.get_final_message()
            usos.append(resp.usage)
            _checar_recusa(resp)
            conteudo_total += list(resp.content)
            if resp.stop_reason != "pause_turn":
                break
            mensagens.append({"role": "assistant", "content": resp.content})
        pesquisa = "\n".join(b.text for b in conteudo_total if b.type == "text").strip()
        fontes = _fontes(conteudo_total)

        # Etapa 2: síntese estruturada
        progresso("Montando cenários e análise estruturada...")
        resp = client.beta.messages.create(
            model=modelo, max_tokens=16000, system=SISTEMA_SINTESE,
            messages=[{"role": "user", "content": (
                f"Ação: {ticker}\nPreço atual: {preco_atual:.2f}\n"
                f"Horizonte dos cenários: {horizonte_desc}\n\n"
                f"## Dados quantitativos e contexto\n{contexto_texto}\n\n"
                f"## Pesquisa na web\n{pesquisa}"
            )}],
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": ESQUEMA}},
            betas=[BETA_FALLBACK], fallbacks="default",
        )
        usos.append(resp.usage)
        _checar_recusa(resp)
        if resp.stop_reason == "max_tokens":
            raise ErroAnalise("A resposta ficou longa demais e foi cortada. Tente novamente.")
        dados = json.loads(_texto(resp))
    except anthropic.AuthenticationError:
        raise ErroAnalise("Chave da API inválida. Confira a chave em console.anthropic.com.")
    except anthropic.PermissionDeniedError:
        raise ErroAnalise("A chave não tem permissão para este modelo ou recurso (busca na web).")
    except anthropic.RateLimitError:
        raise ErroAnalise("Limite de uso da API atingido. Aguarde um pouco e tente de novo.")
    except anthropic.BadRequestError as e:
        msg = e.message
        if "credit" in msg.lower() or "billing" in msg.lower():
            raise ErroAnalise("Sem créditos na conta da API. Adicione créditos em console.anthropic.com.")
        raise ErroAnalise(f"Requisição rejeitada pela API: {msg}")
    except anthropic.APIStatusError as e:
        raise ErroAnalise(f"Erro da API ({e.status_code}). Tente novamente em instantes.")
    except anthropic.APIConnectionError:
        raise ErroAnalise("Sem conexão com a API da Anthropic. Verifique a internet.")

    custo, buscas = _custo(modelo, usos)
    return AnaliseIA(dados, pesquisa, fontes, modelo, custo, buscas, preco_atual)


def montar_contexto_texto(ticker, df, resultados, ctx, horizontes_desc) -> str:
    """Resume em texto tudo o que o app já sabe, para enviar ao analista de IA."""
    close = df["Close"]
    linhas = [
        f"Último fechamento: {close.iloc[-1]:.2f} em {df.index[-1]:%d/%m/%Y}",
        f"Variação 1 mês: {(close.iloc[-1] / close.iloc[-22] - 1) * 100:+.1f}%",
    ]
    if len(close) > 253:
        linhas.append(f"Variação 12 meses: {(close.iloc[-1] / close.iloc[-253] - 1) * 100:+.1f}%")
    linhas.append(f"Volatilidade diária (60d): {close.pct_change().iloc[-60:].std() * 100:.2f}%")

    linhas.append("\nPrevisões do modelo quantitativo (machine learning com indicadores técnicos e macro):")
    for h, r in resultados.items():
        linhas.append(
            f"- {horizontes_desc.get(h, f'{h} pregões')}: {r.preco_previsto:.2f} ({r.retorno_previsto:+.2f}%), "
            f"faixa 80% {r.intervalo_80[0]:.2f}–{r.intervalo_80[1]:.2f}, prob. de alta {r.prob_alta * 100:.0f}%, "
            f"confiabilidade {r.confiabilidade} (acerto de direção no teste {r.acerto_direcao:.1f}%)"
        )
        linhas.append(f"  indicadores mais relevantes: {', '.join(r.importancias.head(5).index)}")

    emp = ctx.get("empresa") or {}
    if emp.get("nome"):
        linhas.append(f"\nEmpresa: {emp['nome']} | setor: {emp.get('setor')} | indústria: {emp.get('industria')}")
    if emp.get("preco_alvo_medio"):
        linhas.append(
            f"Consenso de analistas ({emp.get('n_analistas')} analistas): preço-alvo médio {emp['preco_alvo_medio']}, "
            f"mín {emp.get('preco_alvo_min')}, máx {emp.get('preco_alvo_max')}, recomendação {emp.get('recomendacao')}"
        )
    focus = ctx.get("focus")
    if focus is not None and len(focus):
        linhas.append("\nBoletim Focus (mediana das expectativas):")
        for _, f in focus.iterrows():
            mud = f" (há 4 semanas: {f['Há 4 semanas']})" if f["Há 4 semanas"] == f["Há 4 semanas"] else ""
            linhas.append(f"- {f['Indicador']} {f['Ano']}: {f['Mediana']}{mud}")

    for chave, titulo in (("noticias_empresa", "Manchetes recentes da empresa"),
                          ("noticias_mercado", "Manchetes recentes do mercado")):
        noticias = ctx.get(chave) or []
        if noticias:
            linhas.append(f"\n{titulo}:")
            for n in noticias[:15]:
                data = f"{n.data:%d/%m}" if n.data is not None else "?"
                linhas.append(f"- [{data}] {n.titulo} ({n.fonte})")
    return "\n".join(linhas)
