# umbigo

Backtest de **perp** BTC, ETH e SOL na Hyperliquid. Renda extra. Não é plano de demissão, e esta versão não manda ordem.

Três pastas:

| Pasta | Função |
|---|---|
| `Client/` | API pública de candle (`/info`, sem chave) e SQLite local |
| `Models/` | Modelo estático. O primeiro é TSMOM com vol alvo de 30% e teto de 1x |
| `Esteira/` | Escolhe o parâmetro no treino, congela, testa no período que ele não viu |

Treinar aqui não é rede neural. É uma grade curta de lookback e janela de volatilidade, marcada só no treino. O teste não vota. O artefato fica em `model_artifacts` e uma cópia legível em `Models/artifacts/`.

## Rodar

Na raiz do repositório, Python 3.10+:

```bash
python3 -m unittest discover -s tests -v
python3 -m Esteira.cli run
python3 -m Esteira.cli run --coin BTC --interval 1d
```

Comandos separados: `sync`, `train`, `backtest`. SQLite default: `data/umbigo.sqlite` (não vai pro git).

```bash
sqlite3 data/umbigo.sqlite "SELECT id, coin, interval, params_json FROM model_artifacts"
```

## O que os números não incluem

- **Funding.** Perp segurado paga ou recebe funding. O PnL daqui é preço, taxa de taker (4.5 bp, tier base) e 1 bp de slippage. Se o sharpe mal paga a taxa, funding acaba com ele.
- **Histórico de mentira.** O diário da API começa em 2020 com volume zero. Trade de verdade na HL começa em fev/2023 (SOL um pouco depois). Esse prefixo é cortado.
- **Teto de 5000 candles.** 1h são ~7 meses, 4h ~2 anos. O default é `1d` porque é o intervalo em que o histórico negociável cabe inteiro.
- **Spot.** BTC/ETH/SOL aqui são o perp. Spot na HL usa outro símbolo (`@107` e afins).

Se isso fizer sentido com dinheiro de mentira, o próximo passo é um FastAPI em cima do mesmo client e do mesmo artefato congelado. Ainda sem ordem.
