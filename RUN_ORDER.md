## RUN ORDER (Opdateret)

Denne fil beskriver den anbefalede sekventielle kørsel af pipeline, formål for hvert trin, vigtigste input/outputs og relevante flags.

### Hurtigt overblik (pipeline)

```
aktie.csv ──> fetch_history_pro ──> data_all/ ──> compute_features ──> features.parquet/ ──> make_rebased_windows ──> rebased/ ──> (reference_index) ──> analog_matcher_watch ──> analog_out/
                                                                                               │
                                                                                               └─────────────── benchmarking / prefilter / index
```

Alle hovedscripts skriver en metadatafil `_meta.json` i deres outputmappe med tidsstempel, git commit, konfiguration og versionsinfo.

---
### 1. Hent historik – `fetch_history_pro.py`
Henter daglig OHLCV + (valgfrit) dividender og splits.

Input:
- `aktie.csv` (kolonne: `ticker`)

Standard output (`--preset standard`):
- CSV: `data_all/history_all.csv`
- Partitioneret Parquet: `data_all/history_all_parquet/Ticker=.../*.parquet`
- Actions: `<TICKER>_dividends.*`, `<TICKER>_splits.*`
- Metadata: `data_all/_meta.json`
- Rapport: `data_all/run_report.json`

Eksempel:
```
& "./venv/Scripts/python.exe" ./fetch_history_pro.py --preset standard
```

Vis konfiguration uden kørsel:
```
& "./venv/Scripts/python.exe" ./fetch_history_pro.py --preset standard --show-config
```

Nyttige flags:
- `--incremental` (default i preset) kun tilføj nye datoer
- `--actions` henter udbytter/splits
- `--per-ticker` separate filer
- `--fail-on-empty N` exit 1 hvis >N tomme tickers
- `--log-file run_log.jsonl` struktureret JSON log

---
### 2. Beregn features – `compute_features.py`
Tilføjer MA (20/50/200) og RSI (14) som standard.

Input forsøgsrækkefølge (preset):
1. `data_all/history_all_parquet/`
2. fallback: `data_all/history_all.parquet`

Output:
- Parquet partitioneret: `features.parquet/Ticker=.../*.parquet`
- CSV kopi: `features.csv` (afrundet, kan være stor)
- Metadata: `features.parquet/_meta.json`

Eksempel:
```
& "./venv/Scripts/python.exe" ./compute_features.py --preset standard
```

Konfig check:
```
& "./venv/Scripts/python.exe" ./compute_features.py --preset standard --show-config
```

Nøgleflags:
- `--ma 10 30 90` ændre glidende vinduer
- `--rsi 21` alternative RSI-længde
- `--use-adjclose` (default i preset) beregn på AdjClose

---
### 3. Rebased vinduer – `make_rebased_windows.py`
Genererer vinduer [-before,+after] per (Ticker, RefDate) og rebaser værdier til 100 på offset 0.

Output (preset):
- Per-ticker: `rebased/<TICKER>_rebased.parquet` (+ CSV sample) eller combined.
- Metadata: `rebased/_meta.json`

Vigtige flags:
- `--before 75 --after 25` (standard preset)
- `--drop-rows-if-nan any|all|none` (preset: any)
- `--require-full-window` skip vinduer med tabte rækker (preset: on)
- `--csv-head/--csv-tail` sampling i CSV (preset: 1000/1000)

Eksempel:
```
& "./venv/Scripts/python.exe" ./make_rebased_windows.py --preset standard
```

Konfig check:
```
& "./venv/Scripts/python.exe" ./make_rebased_windows.py --preset standard --show-config
```

---
### 4. (Valgfri) Signatur index – `reference_index.py`
Bygger et kompakt index til prefilter i analog matcher.

Output:
- `rebased_index.parquet`

Eksempel:
```
& "./venv/Scripts/python.exe" ./reference_index.py --rebased-dir rebased --out rebased_index.parquet \
  --lookback 75 --sig-downsample 8 --match-cols AdjClose_Rebased,MA_20_Rebased,MA_50_Rebased,MA_200_Rebased
```

---
### 5. Analog matcher – `analog_matcher_watch.py`
Finder historiske vinduer der ligner det aktuelle (eller feature-baserede) lookback og viser deres efterfølgende udvikling.

Input:
- `watch.csv` (kolonne: `Ticker`)
- `rebased/` og valgfrit `rebased_index.parquet`

Output:
- `analog_out/<TICKER>_scores.csv`
- `analog_out/<TICKER>_futures.parquet` (nu inkl. Lookback/Horizon/MatchCols)
- `analog_out/all_scores.csv`
- (benchmark snapshots i `analog_out/benchmark_runs/`)

Eksempler:
```
& "./venv/Scripts/python.exe" ./analog_matcher_watch.py --watch watch.csv --ref-dir rebased --verbose
& "./venv/Scripts/python.exe" ./analog_matcher_watch.py --watch watch.csv --use-index --index-file rebased_index.parquet \
   --prefilter-topk 500 --sig-downsample 8 --metric mse --score-workers 4 --verbose
```

Nyttige flags:
- `--query-from-features` brug eksisterende features i stedet for live download
- `--use-index` + `--prefilter-topk` prefilter
- `--cache-ref` + `--cache-ref-size` filcache
- `--metric mse|corr` + `--min-corr` (filter når corr)
- `--score-workers N` parallel scoring
- `--dry-run` simulation uden netværk/skrivning
- `--heartbeat-interval 5` fremdriftslinjer

Show config:
```
& "./venv/Scripts/python.exe" ./analog_matcher_watch.py --watch watch.csv --show-config
```

---
### 6. Benchmarking – `scripts/benchmark_prefilter.py`
Sammenligner forskellige matching modes (full, index, index-vectorized).

Eksempel:
```
& "./venv/Scripts/python.exe" ./scripts/benchmark_prefilter.py --mode all --metric mse --topk 400 --score-workers 4
```

Output: JSON result (specificeret via `--json-out` eller auto under `analog_out/benchmark_runs/`).

---
### VS Code Tasks & One-click

- Task: "Fetch: preset standard" → fetch step
- Task: "Features: preset standard" → features step
- Task: "Rebased: preset standard" → rebased step
- Task: "Run pipeline (presets)" → fetch → features sekventielt

Kør via: `Ctrl+Shift+P` → "Run Task".

---
### Metadata filer (`_meta.json`)
Alle tre pipeline-trin (fetch, features, rebased) skriver en `_meta.json` med fx:
```json
{
  "component": "features",
  "timestamp": "2025-09-25T05:31:10Z",
  "git_commit": "<hash>",
  "python": "3.13.6",
  "pandas": "2.3.2",
  "config": { ... }
}
```
Brug dem til reproducérbarhed og audit.

---
### Fejlfinding (kort)
- Ingen output? Tjek `--show-config` for stier.
- Manglende Parquet læsning: versionskonflikt mellem pandas og pyarrow.
- Tomme match resultater: Tjek at `lookback` matcher rebased vinduers lookback (default 75).
- Lav performance i matcher: aktiver index + prefilter + øg `--score-workers`.

---
### Opsummering af hovedkommandoer
```
& "./venv/Scripts/python.exe" fetch_history_pro.py --preset standard
& "./venv/Scripts/python.exe" compute_features.py --preset standard
& "./venv/Scripts/python.exe" make_rebased_windows.py --preset standard
& "./venv/Scripts/python.exe" reference_index.py --rebased-dir rebased --out rebased_index.parquet
& "./venv/Scripts/python.exe" analog_matcher_watch.py --watch watch.csv --use-index --index-file rebased_index.parquet --verbose
```

---
### Fremtidige udvidelser (idéer)
- Central konfigfil (TOML) for hele pipeline.
- Ekstra indikatorer som plugins.
- Metrics eksport til dashboard.
