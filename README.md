# Aktier v2

Et robust Python-script til at hente daglige aktiekurser (1d) fra Yahoo Finance med batch-download, retries, incremental updates, actions (udbytter/splits), UTC-normalisering, logging og rapport.

## Krav / Setup

1. Opret virtuelt miljø og installer pakker

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Inputfil

- `aktie.csv` med kolonnen `ticker`:

```
ticker
AAPL
MSFT
NOVO-B.CO
```

## Kørsel

Grundlæggende:

```powershell
python fetch_history_pro.py --input aktie.csv --out data
```

Nyttige options:
- `--per-ticker` Gem en fil pr. ticker
- `--incremental` Hent kun nye datoer (byg på eksisterende filer)
- `--actions` Gem udbytter/splits
- `--adjusted-only` Brug AdjClose som Close
- `--batch-size N` Batchstørrelse (default 30)
- `--float-dp N` Antal decimaler i CSV (Parquet bevarer fuld præcision)
- `--compression snappy|zstd|gzip` Parquet-kompression
- `--partition-by Ticker` Partitionér samlet Parquet efter kolonne (kun samlet tilstand)
- `--validate-only` Valider input/net uden download
- `--dry-run` Download i RAM uden at skrive til disk
- `--only AAPL,MSFT` Begræns til udvalgte tickers
- `--fail-on-empty N` Exit 1 hvis >N tomme tickers
- `--log-file path` JSON-lines logfil

## Output

- Per ticker: `{TICKER}.csv` og `{TICKER}.parquet`
- Samlet: `history_all.csv` og `history_all.parquet` eller mappe `history_all_parquet/` ved partitionering
- Actions: `{TICKER}_dividends.*` og `{TICKER}_splits.*`
- Rapport: `run_report.json`

## Bemærkninger
- CSV afrundes til `--float-dp` decimaler. Parquet bevarer fuld præcision.
- Yahoo Finance har rate limits. Scriptet kører med batches, retries og korte pauser.

## Licens
MIT
