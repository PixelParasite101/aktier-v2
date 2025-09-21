# Projekt TODO

En samlet oversigt over igangværende og planlagte opgaver for aktier-v2. Status opdateres løbende.

## Statuslegende
- [x] færdig
- [ ] ikke startet
- [~] i gang

## Færdige opgaver
- [x] Commit og push den nuværende kode (kun kode/scripts/tests)
- [x] Analog matcher: query fra features (`--query-from-features`, `--query-features-dir`)
- [x] Parquet kolonneprojektion ved reference-scan (læser kun nødvendige kolonner)

## Backlog

1) Ydelse og skalerbarhed
- [ ] Paralleliser reference-scanning
  - Brug `concurrent.futures`/`multiprocessing` til at scanne pr. fil/partition parallelt.
  - Accept: Målelig speedup på mellemliggende datasæt; kontrolleret RAM-forbrug.
- [ ] Indeks/cache af referencevinduer
  - Byg let indeks (CSV/Parquet/SQLite) over `Ticker, RefDate-range, fil` for at undgå fulde reskanninger.
  - Accept: Matcher læser maks ét fil-hit pr. match i gennemsnit for typiske opslag.

2) Matchkvalitet og fleksibilitet
- [ ] Saml scoring og weights
  - CLI for vægte/kolonnevalg (fx `--weights price=1,ma=0.5,rsi=0.2`).
  - Accept: Samme topN i standardkonfiguration; tydelig effekt ved ændrede vægte.

3) Observability
- [ ] Logging og metrics
  - Tidsmålinger, antal kandidater, filtreringer, topN-diagnostik; output til JSON/CSV.
  - Accept: En JSON-fil per kørsel med varigheder og kandidat-antalsstatistikker.

4) Korrekthed og robusthed
- [ ] End-to-end analog smoke test (features-mode)
  - Syntetiske features → rebased → analog matcher `--query-from-features` → deterministisk topN.
  - Accept: Test kører grønt lokalt (pytest) og dækker basisflowet.

5) DX og dokumentation
- [ ] VS Code tasks til matcher
  - Tasks for både yfinance og features-mode med presets.
  - Accept: Ét klik/kommando for at køre matcher i begge modes.
- [ ] Analog matcher preset
  - `--preset standard` for `analog_matcher_watch.py` med paths, lookback/horizon og fornuftige default-kolonner.
  - Accept: Kørsel uden flags virker med projektets defaults.
- [ ] Docs/README-opdateringer
  - Dokumentér matcher-flags, CSV head/tail sampling, anbefalet Parquet-first workflow.
  - Accept: README-sektion med korte “Try it”-eksempler (PowerShell).
- [ ] Line-endings normalisering
  - `.gitattributes` der normaliserer LF og markerer binære filer korrekt.
  - Accept: Ingen CRLF-advarsler ved `git add`.
- [ ] Type hints og linting
  - Udvid type hints i public funktioner (compute_features, make_rebased_windows, analog matcher).
  - Accept: Ruff passerer; (valgfrit) mypy uden fejl.

6) Små forbedringer
- [ ] Timezone-sikker fetch-rapport
  - Udskift `datetime.utcnow()` med timezone-aware UTC i `fetch_history_pro.py`.
  - Accept: Ingen deprecation-warning.

## Hurtige noter
- Autoritativt format: Parquet. CSV bruges til preview (head/tail sampling i rebased CSV).
- Rebased vinduer: `--require-full-window` og `--drop-rows-if-nan any` anbefales (preset standard).
- Matcher kan køre offline ved `--query-from-features` og læser kun nødvendige kolonner fra rebased.

## Forslag fra kodegennemgang

Nedenstående er konkrete, lavrisiko opgaver jeg anbefaler efter en kort gennemgang af projektet
(tests kørte grønt lokalt). Prioriter efter tid/nytte:

- [ ] Tilføj CI (GitHub Actions) der kører `pip install -r requirements.txt` og `pytest` på push/pr.
  - Accept: Tests kører automatisk og fejler hvis et commit breaker noget.
- [ ] Optimer `scripts/build_rebased_index.py` så den ikke læser samme parquet-fil to gange ved kolonneprojektion.
  - Accept: Én læsning per fil ved projektion (brug pyarrow schema eller read with columns fallback én gang).
- [ ] Tilføj en kort "Developer" sektion i `README.md` med instruktioner for at oprette venv og køre tests.
  - Accept: Enkle copy/paste-kommandoer (PowerShell) som nye bidragydere kan bruge.
- [ ] Overvej at gemme downsample signature i parquet som en list/array-typed kolonne i stedet for JSON-string,
  hvis signaturen skal bruges ofte uden json-parsing.
  - Accept: Hurtigere læs og enklere queries i index-workflow.
- [ ] Tjek og noter kompatibilitet mellem `pandas` og `pyarrow` i `requirements.txt` (opgraderings-note).

Hvis du vil, kan jeg lave en PR med 1) en GitHub Actions workflow og 2) den `build_rebased_index.py`-optimering.

## Prioriteret handlingsliste (foreslået)

Her er en konkret handlingsliste, prioriteret efter hurtig nytte og indsats. Hvert punkt har en kort forklaring, estimeret indsats og et foreslået next-step.

Hurtige wins (lav indsats)

1) README: Developer-sektion
  - Hvad: Tilføj quick-start (opret venv, installer, kør tests, kør presets).
  - Estimat: 10–20 min.
  - Next: Tilføj kort afsnit i `README.md`.

2) CI: statusbadge i README
  - Hvad: Tilføj badge for GitHub Actions workflow i `README.md`.
  - Estimat: 5–10 min.
  - Next: Indsæt badge-markup (efter CI er aktiv).

3) Requirements-compatibility note
  - Hvad: Kort note om pandas/pyarrow kompatibilitet (og evt. anbefalede versioner).
  - Estimat: 10–30 min.
  - Next: Tjek aktuelle versioner og tilføj note i README eller `requirements.txt`.

Medium indsats (værdifulde forbedringer)

4) Gem signature som native Parquet-list/array
  - Hvad: Gem `*_sig` som list/array i stedet for JSON-string for hurtigere læsning.
  - Estimat: 1–2 timer + tests.
  - Next: Prototype skriv/read af én fil med array-kolonne og test index-workflow.

5) Logging improvements (fetch_history_pro)
  - Hvad: Udvid log-event med fejldetaljer/stacktraces og durations pr. ticker.
  - Estimat: 30–60 min.
  - Next: Tilføj richer exception-logging til `log_event`.

Lavere prioritet / større ændringer

6) Parallel scanning / indeks-performance
  - Hvad: Brug `concurrent.futures` til parallelt læs af rebased filer ved index-build.
  - Estimat: 2–4 timer.
  - Next: Prototype med begrænset worker-count og måling.

7) Matcher weights CLI
  - Hvad: CLI for at konfigurere vægte og kolonnevalg (fx `--weights price=1,ma=0.5`).
  - Estimat: 1–3 timer.
  - Next: Tilføj parser og tests.

8) End-to-end smoke-test
  - Hvad: Test der kører syntetisk data gennem features→rebased→index→matcher.
  - Estimat: 1–2 timer.
  - Next: Implementér pytest-case med små syntetiske datasæt.

Repo-hygiejne

9) `.gitattributes` for line endings
  - Hvad: Tilføj file for LF-normalisering og binære filer.
  - Estimat: 10–15 min.
  - Next: Opret `.gitattributes` med anbefalede regler.

10) Type hints og linting
  - Hvad: Tilføj eller udvid type hints i public funktioner; kør ruff.
  - Estimat: 1–3 timer (udvalgte filer først).
  - Next: Start med `compute_features.py` og `make_rebased_windows.py`.

Sig hvilke punkter (fx “1–3” eller “4 og 6”) jeg skal implementere nu, så går jeg i gang og opdaterer `TODO.md` løbende.