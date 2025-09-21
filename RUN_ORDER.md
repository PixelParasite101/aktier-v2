Kort kørsel-vejledning for projektet

Formål: Beskriv hvilken rækkefølge de vigtigste scripts skal afvikles i, hvilke filer de læser fra, og hvilke filer/mappe de skriver.

1) Hent historik

- Script: `fetch_history_pro.py`
- Hvad det gør: Henter daglig OHLCV + actions (dividender/splits) fra Yahoo Finance for tickers i `aktie.csv`.
- Input: `aktie.csv` (kolonne: `ticker`)
- Output (default preset=standard):
  - CSV: `data_all/history_all.csv` (samlet)
  - Parquet (partitioneret): `data_all/history_all_parquet/Ticker=.../*.parquet`
  - Per-ticker filer (hvis `--per-ticker`): `data_all/<TICKER>.parquet`, plus `<TICKER>_dividends.parquet` osv.
- Eksempel (PowerShell):

  & "./venv/Scripts/python.exe" "./fetch_history_pro.py" --preset standard

2) Beregn features

- Script: `compute_features.py`
- Hvad det gør: Beregner MA'er (f.eks. 20/50/200) og RSI pr. ticker.
- Input (preset=standard forsøger automatisk): `data_all/history_all_parquet` eller `data_all/history_all.parquet`
- Output (default preset=standard):
  - Partitioneret Parquet: `features.parquet/Ticker=.../*.parquet`
  - CSV-kopi: `features.csv` (kan være stor)
- Eksempel:

  & "./venv/Scripts/python.exe" "./compute_features.py" --preset standard

3) Lav rebased vinduer (reference)

- Script: `make_rebased_windows.py`
- Hvad det gør: For hver ticker og reference-dato laver det vinduer [-before,+after] (fx -75..+25) hvor værdier rebases mod dag 0 (ref-dato). Dette er reference-datasættet brugt af analog matcher.
- Input: `features.parquet` (eller `features.csv`) — preset=standard bruger `features.parquet` automatisk
- Output (default preset=standard):
  - Per-ticker Parquet/CSV i `rebased/` (fx `rebased/AAPL_rebased.parquet`) eller samlet `rebased_all.parquet` afhængig af flags
- Eksempel:

  & "./venv/Scripts/python.exe" "./make_rebased_windows.py" --preset standard

4) Analog matcher (watch)

- Script: `analog_matcher_watch.py`
- Hvad det gør: For tickers i `watch.csv` bygger den en query (fra `features.parquet` eller yfinance), rebaser den, og sammenligner mod alle referencevinduer i `rebased/` for at finde topN matches. Gemmer scores og fremtidsdata (futures).
- Input: `watch.csv` (skal have header `Ticker`), `rebased/` (output fra step 3)
- Output:
  - Per-query scores CSV: `analog_out/<TICKER>_scores.csv`
  - Per-query futures Parquet: `analog_out/<TICKER>_futures.parquet`
  - Samlet oversigt: `analog_out/all_scores.csv` (hvis flere queries)
- Kør fra VS Code Run (jeg har lavet en launch-konfig med `--verbose`):
  - Vælg "Run analog matcher (watch)" i Run & Debug og tryk Run
- Eller kør i terminal:

  & "./venv/Scripts/python.exe" "./analog_matcher_watch.py" --watch "./watch.csv" --ref-dir "./rebased" --verbose

  Yderligere nyttige flags (nyere tilføjelser):

  - `--show-config` — Vis den opsamlede, resolved konfiguration og afslut (godt til debugging eller CI).
  - `--config-out <file>` — Skriv den resolved konfiguration til en JSON-fil.
  - `--log-file <file>` — Skriv strukturerede JSON-log-linjer til en fil ud over console-log.
  - `--heartbeat-interval <seconds>` — Udskriv en kort "jeg lever"-linje hvert N. sekunder så du kan se processen arbejder (0=deaktiver).
  - `--dry-run` — Kør scriptet i simuleret tilstand (ingen netværkskald eller filskrivninger). Nyttigt til at teste at systemet kører og at heartbeats vises.

  Eksempel — kør analog matcher i simuleret tilstand med hyppige heartbeats og skriv en JSON-log:

    & "./venv/Scripts/python.exe" "./analog_matcher_watch.py" --watch "./watch.csv" --dry-run --heartbeat-interval 2 --log-file run_report.json

Kort om flowet

- fetch_history_pro -> data_all/ (history)
- compute_features -> features.parquet/
- make_rebased_windows -> rebased/
- analog_matcher_watch -> analog_out/

Bemærkninger

- Parquet-filer bevares i fuld numerisk præcision; CSV'er afrundes til `--float-dp` (typisk 4). 
- Mange scripts har en `--preset standard` som sætter fornuftige default-stier.
- Hvis du vil se runtime-output i VS Code, brug den nye Run-konfiguration "Run analog matcher (watch)" (sender `--verbose`).

Hvis du vil, kan jeg også tilføje et lille PowerShell-skript `run_pipeline.ps1` med alle fire skridt i rækkefølge (med bekræftelse mellem skridt).