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

Standard kørsel (presets):

```powershell
# Hent data med gode defaults (incremental, actions, partitioneret Parquet pr. Ticker)
python fetch_history_pro.py --preset standard

# Beregn features fra standard-output og skriv både Parquet (partitioneret) og CSV
python compute_features.py --preset standard

# Lav rebased vinduer [-75,+25] med preset standard
python make_rebased_windows.py --preset standard
```

Andre presets:
- `python fetch_history_pro.py --preset fast`  (hurtig: incremental, uden actions, mindre batch)
- `python fetch_history_pro.py --preset validate`  (tjek input/netværk uden download)

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
 - `--preset standard|fast|validate` Forudindstillede kørsler (kan overrides af øvrige flags)
- `--csv-head N` og `--csv-tail N` (valgfrit) — skriv kun de første N og/eller sidste N rækker til CSV. Parquet skrives altid komplet.

## Output

- Per ticker: `{TICKER}.csv` og `{TICKER}.parquet`
- Samlet: `history_all.csv` og `history_all.parquet` eller mappe `history_all_parquet/` ved partitionering
- Actions: `{TICKER}_dividends.*` og `{TICKER}_splits.*`
- Rapport: `run_report.json`

## Bemærkninger
- CSV afrundes til `--float-dp` decimaler. Parquet bevarer fuld præcision.
- CSV kan begrænses med `--csv-head/--csv-tail` for hurtig inspektion (fx 1000/1000). Brug Parquet for fuld data.
- Yahoo Finance har rate limits. Scriptet kører med batches, retries og korte pauser.

### Preset-detaljer
- fetch_history_pro.py `--preset standard` sætter som udgangspunkt:
	- `--input aktie.csv`, `--out data_all`, `--incremental`, `--actions`, `--partition-by Ticker`, `--batch-size 30`, `--compression snappy`, `--float-dp 4` (kun CSV)
	- Du kan override med egne flags (fx ændre `--out` eller `--float-dp`).
- compute_features.py `--preset standard` sætter som udgangspunkt:
	- `--input data_all/history_all_parquet` (falder tilbage til `data_all/history_all.parquet` hvis den findes),
		`--out features.parquet`, `--partition-by Ticker`, `--csv features.csv`, `--use-adjclose`, `--float-dp 4` (kun CSV)
	- Du kan override med egne flags.

- make_rebased_windows.py `--preset standard` sætter som udgangspunkt:
	- `--input features.parquet` (mappe eller fil), `--out rebased`, `--before 75`, `--after 25`, `--per-ticker`, `--float-dp 4`, `--format both`, `--csv-head 1000`, `--csv-tail 1000`, `--drop-rows-if-nan any`
	- Vinduer genereres pr. Ticker for hver reference-dato. Rækker hvor alle rebased-værdier er NaN (typisk ikke-handelsdage) filtreres fra.
	- Output-kolonneorden: `Ticker, RefDate, Offset, Date, AdjClose_Rebased, Close_Rebased, MA_20_Rebased, MA_50_Rebased, MA_200_Rebased, RSI_*`

### Kør uden argumenter (Run Python File)

Alle tre scripts default’er til `--preset standard`, når de køres uden argumenter (fx via VS Code “Run Python File”):
- `fetch_history_pro.py` → standard preset
- `compute_features.py` → standard preset
- `make_rebased_windows.py` → standard preset

Du kan stadig override alle værdier ved at angive flags eksplicit.

## VS Code: kør uden terminal

I mappen `.vscode/` findes tasks:
- “Fetch: preset standard”
- “Features: preset standard”
- “Run pipeline (presets)” (kører Fetch → Features i rækkefølge)
- “Rebased: preset standard”

Åbn kommandopalletten (Ctrl+Shift+P) → “Run Task” og vælg den ønskede task.

## Rebased vinduer

Scriptet `make_rebased_windows.py` genererer vinduer på [-before, +after] handelsdage omkring en reference-dato og rebaserer værdier til 100 på ref-dagen.

Eksempel med standard preset:

```powershell
python make_rebased_windows.py --preset standard
```

Nyttige flags:
- `--before 75 --after 25` (ændr løsningens vinduesstørrelse)
- `--per-ticker` (en fil pr. ticker; default i preset standard)
- `--format csv|parquet|both`
- `--drop-rows-if-nan none|any|all` (default i preset: any). Styr rækker med NaN i rebased-kolonner:
	- `none` behold alle rækker
	- `any` drop rækker hvor mindst én rebased-kolonne er NaN (preset)
	- `all` drop rækker hvor alle rebased-kolonner er NaN
- `--float-dp N` (afrund kun CSV; Parquet fuld præcision)

Filtrering og kvalitet:
- Input deduplikeres på (Ticker, Date) og sorteres.
- Reference-rækker med 0/NaN i reference-værdier springes over (undgår division med 0).
- Rækker hvor alle rebased-værdier er NaN (typisk ikke-handelsdage) filtreres fra i output.

## Valgfrie “one-click” scripts

`run_pipeline.ps1` og `run_pipeline.bat` er med for nem kørsel uden for VS Code (dobbeltklik, Task Scheduler). De er valgfrie, da
VS Code tasks og scripts’ no-args default til `--preset standard` dækker de fleste behov.

## Licens
MIT

## Developer

CI status: ![CI](https://github.com/PixelParasite101/aktier-v2/actions/workflows/ci.yml/badge.svg)

Quick start (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Run tests:

```powershell
python -m pytest -q
```

Run the standard pipeline locally (quick):

```powershell
python fetch_history_pro.py --preset standard
python compute_features.py --preset standard
python make_rebased_windows.py --preset standard
```

Compatibility note:

- `pandas` and `pyarrow` versions must be compatible for Parquet I/O. If you see parquet read/write errors after upgrading packages, try pinning `pandas` to the version used here and upgrading/downgrading `pyarrow` accordingly. The project was tested with the versions in `requirements.txt`.

