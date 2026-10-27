# Previsão de Ações

App em Python que prevê o preço de ações combinando três coisas:

1. **Modelo quantitativo** (machine learning) treinado com o histórico da ação e dados externos:
   Ibovespa/S&P 500, dólar, Selic, VIX, petróleo, **expectativas de inflação do boletim Focus** e
   **índice de incerteza política e econômica (EPU Brasil)**.
2. **Notícias e contexto**: manchetes recentes com tom positivo/negativo e marcação de temas políticos,
   consenso de analistas e boletim Focus.
3. **Análise com IA (opcional)**: o Claude pesquisa na web o cenário político, o mercado e as especulações
   atuais e monta cenários otimista, base e pessimista com probabilidades.

O app guarda cada previsão e, quando a data chega, mostra **quem acertou mais** (aba 🎯 Acompanhamento).

## Instalação

```bash
pip install -r requirements.txt
```

## Usar

Dois cliques em `Abrir Previsao de Acoes.bat` (Windows) ou:

```bash
streamlit run app.py
```

## Análise com IA

Precisa de uma chave da API da Anthropic:

1. Crie uma conta em https://console.anthropic.com e adicione créditos (cobrança por uso).
2. Em **API Keys**, crie uma chave.
3. No app, abra **🤖 Análise com IA** na barra lateral, cole a chave e clique em **Lembrar chave neste computador**.

Custo aproximado: US$ 0,15 a 0,60 por análise com o Claude Opus 5.5, metade com o Sonnet 5.5.

A chave fica salva em `.streamlit/secrets.toml`, na pasta do app. Não compartilhe esse arquivo.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `app.py` | Interface (Streamlit) |
| `motor_previsao.py` | Dados, indicadores, otimização, validação e previsão |
| `contexto.py` | Notícias, sentimento, consenso de analistas e Focus |
| `analista_ia.py` | Análise com o Claude (busca na web + cenários) |
| `historico.py` | Histórico de previsões e placar de acertos |
| `previsao_acoes.py` | Versão de linha de comando (só o modelo quantitativo) |

> Ferramenta educacional. Previsões de mercado são altamente incertas. Não é recomendação de investimento.
