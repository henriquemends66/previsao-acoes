"""
Venda de pacotes de análises com IA via Pix (API do Mercado Pago).

Fluxo:
1. `criar_pix` gera uma cobrança Pix e um código de acesso, gravado no próprio pagamento
   (campo external_reference). O visitante recebe o QR Code e o código.
2. `validar` confirma o código: se o pagamento estiver aprovado, o código vale N análises.
3. `usar` desconta uma análise; `devolver` estorna a análise se ela falhar.

O Mercado Pago é a fonte da verdade sobre o pagamento; o arquivo creditos.json guarda só
quantas análises cada código já usou. Se o arquivo se perder (ex.: reinício do servidor na nuvem),
o código continua válido e o contador recomeça.
"""

from __future__ import annotations

import json
import secrets
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

API = "https://api.mercadopago.com/v1/payments"
ARQUIVO = Path(__file__).with_name("creditos.json")
ALFABETO = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # sem 0/O, 1/I, para não confundir
_trava = threading.Lock()


class ErroPagamento(Exception):
    pass


@dataclass
class CobrancaPix:
    pagamento_id: int
    codigo: str
    copia_e_cola: str
    qr_base64: str
    link: str | None
    expira_em: pd.Timestamp


def novo_codigo() -> str:
    c = "".join(secrets.choice(ALFABETO) for _ in range(8))
    return f"{c[:4]}-{c[4:]}"


def normalizar_codigo(codigo: str) -> str:
    c = "".join(ch for ch in codigo.upper() if ch.isalnum())
    return f"{c[:4]}-{c[4:]}" if len(c) == 8 else codigo.strip().upper()


def _req(metodo: str, url: str, token: str, **kw):
    try:
        r = requests.request(metodo, url, headers={"Authorization": f"Bearer {token}", **kw.pop("headers", {})},
                             timeout=30, **kw)
    except requests.exceptions.RequestException:
        raise ErroPagamento("Não foi possível conectar ao Mercado Pago. Tente novamente.")
    if r.status_code == 401:
        raise ErroPagamento("Token do Mercado Pago inválido (configuração do dono do app).")
    if r.status_code >= 400:
        try:
            msg = r.json().get("message", r.text[:200])
        except ValueError:
            msg = r.text[:200]
        raise ErroPagamento(f"O Mercado Pago recusou a operação: {msg}")
    return r.json()


# --------------------------------------------------------------------------- #
# Mercado Pago
# --------------------------------------------------------------------------- #
def criar_pix(token: str, email: str, valor: float, n_analises: int, minutos_validade: int = 30) -> CobrancaPix:
    codigo = novo_codigo()
    expira = (pd.Timestamp.now(tz="America/Sao_Paulo") + pd.Timedelta(minutes=minutos_validade)).floor("s")
    corpo = {
        "transaction_amount": round(float(valor), 2),
        "description": f"Pacote de {n_analises} análises com IA - Previsão de Ações",
        "payment_method_id": "pix",
        "payer": {"email": email},
        "external_reference": codigo,
        "date_of_expiration": expira.isoformat(timespec="milliseconds"),  # ex.: 2026-10-27T18:30:00.000-03:00
    }
    dados = _req("POST", API, token, json=corpo, headers={"X-Idempotency-Key": str(uuid.uuid4())})
    tx = (dados.get("point_of_interaction") or {}).get("transaction_data") or {}
    if not tx.get("qr_code"):
        raise ErroPagamento("O Mercado Pago não devolveu o código Pix. Verifique se a conta tem uma chave Pix cadastrada.")
    return CobrancaPix(dados["id"], codigo, tx["qr_code"], tx.get("qr_code_base64", ""), tx.get("ticket_url"), expira)


def status_pagamento(token: str, pagamento_id: int) -> str:
    """approved, pending, rejected, cancelled, refunded..."""
    return _req("GET", f"{API}/{pagamento_id}", token).get("status", "desconhecido")


def buscar_por_codigo(token: str, codigo: str) -> dict | None:
    """Pagamento aprovado mais recente com este código (ou None)."""
    dados = _req("GET", f"{API}/search", token,
                 params={"external_reference": codigo, "sort": "date_created", "criteria": "desc"})
    aprovados = [p for p in dados.get("results", []) if p.get("status") == "approved"]
    return aprovados[0] if aprovados else None


# --------------------------------------------------------------------------- #
# Créditos
# --------------------------------------------------------------------------- #
def _carregar(arquivo: Path) -> dict:
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _salvar(dados: dict, arquivo: Path):
    tmp = arquivo.with_suffix(".tmp")
    tmp.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(arquivo)


def validar(token: str, codigo: str, n_analises: int, arquivo: Path = ARQUIVO) -> int:
    """Retorna quantas análises restam no código. Levanta ErroPagamento se o código não for válido."""
    codigo = normalizar_codigo(codigo)
    with _trava:
        dados = _carregar(arquivo)
        if codigo in dados:
            return dados[codigo]["total"] - dados[codigo]["usados"]
    pag = buscar_por_codigo(token, codigo)
    if pag is None:
        raise ErroPagamento("Código não encontrado ou pagamento ainda não confirmado.")
    with _trava:
        dados = _carregar(arquivo)
        dados.setdefault(codigo, {"pagamento_id": pag["id"], "total": n_analises, "usados": 0,
                                  "ativado_em": str(pd.Timestamp.now().floor("s"))})
        _salvar(dados, arquivo)
        return dados[codigo]["total"] - dados[codigo]["usados"]


def ativar(codigo: str, pagamento_id: int, n_analises: int, arquivo: Path = ARQUIVO):
    """Registra um código cujo pagamento acabou de ser confirmado."""
    with _trava:
        dados = _carregar(arquivo)
        dados.setdefault(codigo, {"pagamento_id": pagamento_id, "total": n_analises, "usados": 0,
                                  "ativado_em": str(pd.Timestamp.now().floor("s"))})
        _salvar(dados, arquivo)


def usar(codigo: str, arquivo: Path = ARQUIVO) -> int:
    """Desconta uma análise e retorna quantas restam. Chame `validar` antes."""
    codigo = normalizar_codigo(codigo)
    with _trava:
        dados = _carregar(arquivo)
        c = dados.get(codigo)
        if c is None or c["usados"] >= c["total"]:
            raise ErroPagamento("Este código não tem mais análises disponíveis.")
        c["usados"] += 1
        _salvar(dados, arquivo)
        return c["total"] - c["usados"]


def devolver(codigo: str, arquivo: Path = ARQUIVO):
    """Estorna uma análise (usada quando a análise falha)."""
    codigo = normalizar_codigo(codigo)
    with _trava:
        dados = _carregar(arquivo)
        if codigo in dados and dados[codigo]["usados"] > 0:
            dados[codigo]["usados"] -= 1
            _salvar(dados, arquivo)
