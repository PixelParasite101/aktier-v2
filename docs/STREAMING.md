# Streaming / Per-file processing design

Short summary
--------------
Dette dokument beskriver en "streaming"-tilgang hvor pipeline-stadierne (fetch -> features -> rebased -> flat_vectorize)
kan køre per input-fil eller per-ticker i stedet for at samle hele datasættet i hukommelsen. Målet er at gøre pipeline'en
skalerbar for mange eller store CSV-filer ved at behandle og opdatere canonical Parquet inkrementelt.

Hvorfor
-------
- Undgå OOM når inputdata ikke kan holdes i RAM.
- Muliggør inkrementel opdatering: procesér kun ny data og append/merge til eksisterende Parquet-filer.
- Gør pipeline mere driftbar (genkør kun for nye inputfiler eller specifikke tickers).

Hurtigt resume af løsningen
---------------------------
- Processér input fil-for-fil.
- For hver ticker i en inputfil: læs eksisterende canonical Parquet (hvis nødvendig), concat med ny data, beregn indikatorer (MA/RSI) over den samlede serie, og skriv atomisk tilbage til per-ticker Parquet.
- Kør rebased-vinduer per-ticker og append nye (Ticker,RefDate,Offset)-rækker til per-ticker rebased-parquet (brug eksisterende append helper hvis kompatibel).
- Flad-vectorize per-ticker rebased output og append til per-ticker flat_vectors eller et samlet index.

Design - Stage-by-stage
-----------------------
1) fetch_history_pro (input: CSV med tickers)
   - For hver inputfil:
     - Læs tickers fra CSV.
     - For hver ticker:
       - hent nye priser (eller hele history hvis ikke incremental), normaliser via `normalize_prices`.
       - hvis `--per-ticker`: skriv `out/{Ticker}.parquet` atomically (tmp -> os.replace).
       - hvis samlet: skriv partitioned Parquet `out/history_all_parquet/` partitioned by `Ticker`.
   - Skriv sampled CSV (fra parquet) efter skriv.

2) compute_features
   - For hver inputfil:
     - læs fil og groupby Ticker.
     - for hver ticker:
       - læs `features/{Ticker}.parquet` hvis exists (for kontinuitet ved indikatorer).
       - concat(existing, new) -> drop_duplicates(subset=['Ticker','Date']) -> sort by Date.
       - re-compute MA/RSI over fuld serie.
       - atomisk skriv `features/{Ticker}.parquet` (tmp -> os.replace).

   - Tests: verifikér at output er identisk med baseline (concat alt og kør non-streaming).

3) make_rebased_windows
   - For hver ticker hvor features blev opdateret:
     - læs `features/{Ticker}.parquet`.
     - kør `make_windows_for_ticker(...)` og producer `rebased` DataFrame.
     - append missing rows til `rebased/{Ticker}_rebased.parquet` ved hjælp af `_append_missing_rows_to_parquet`.
     - opdater sidecar `.meta.json` hvis rewrite.

4) flat_vectorize
   - For hver ticker hvor rebased blev opdateret:
     - læs `rebased/{Ticker}_rebased.parquet` (eller bare de nye rows hvis du gemmer et small index of new rows).
     - flad vinduer til vektorer og append til `flat_vectors/{Ticker}.parquet` eller samlet `flat_vectors.parquet` via append helper.

Atomic writes og lås
--------------------
- Brug tmp-filer og `os.replace(tmp, target)` for at sikre atomisk swap.
- For concurrency: brug per-ticker lockfiler (simple) eller `portalocker` for cross-platform lås.
- Undgå at to processer skriver samme ticker samtidig; orchestrér med en worker-queue hvis parallelisering ønskes.

Append vs full rewrite
----------------------
- Genbrug `_append_missing_rows_to_parquet` når schema matcher. Hvis returnerer -1 -> rewrite hele filen.
- Gem sidecar `.meta.json` med config for append-compat.

Fejlhåndtering og retry
-----------------------
- Log per-ticker fejl til run_report.json.
- Brug retries for fetch (tenacity allerede anvendt).
- Hold tmp-filer når fejl opstår for post-mortem.

Test-plan
---------
- Unit: to små CSV inputfiler (delte tickers) -> kør streaming compute_features og sammenlign features/{Ticker}.parquet med baseline.
- Integration: kør hele pipeline på sample data og verificer counts + checksums af Parquet outputs.

Pickup checklist (hvad skal laves for at implementere)
-----------------------------------------------------
- [ ] Tilføj helper i `utils/common.py` for: expand_input_paths(inputs: str|list|glob) -> list[path]
- [ ] Opdater `src/compute_features.py` til per-file streaming mode (ny flag: `--streaming` eller `--per-file`).
- [ ] Implementér atomic write helper `utils/common.atomic_replace(path, df, **to_parquet_kwargs)`.
- [ ] Brug existing `_append_missing_rows_to_parquet` i `make_rebased_windows.py` for per-ticker append.
- [ ] Tilføj simple per-ticker lock (utils/lock.py) eller brug `portalocker`.
- [ ] Tests: `tests/test_streaming_compute_features.py` + integration smoke test.

Tidsestimat
-----------
- Minimal (compute_features streaming): ~1-2 timer (inkl. tests)
- Full chain streaming (compute -> rebased -> flat): ~4-6 timer (inkl. tests and minor refactors)

Konklusion
----------
Denne løsning gør pipeline robust for store inputmængder og forenkler drift ved at gøre updates inkrementelle og lokale per-ticker. Den bevarer Parquet som canonical format og genbruger allerede tilstedeværende append/sidecar logik i repoet.

---
Dokumentet er gemt som `docs/STREAMING.md` i repoet for nem genoptagelse. Når du er klar implementerer jeg efter prioritet (start med compute_features streaming som minimum).