Målbillede for performance
Ambition: Reducér wall-clock tid pr. query-ticker fra “X sekunder/minutter” til en faktor 5–25 hurtigere afhængigt af datasetstørrelse ved:

Prefilter (signatur-distance) -> reducer kandidater fra N (alle vinduer) til K (f.eks. 300) før fuld scoring.
Batch-læsning/kolonne-projektion kun for kandidater.
Caching + evt. memory-map af index.
NumPy/Numba for scoring (MSE/Corr) i vektorer over kandidater.
Mulig Polars for hurtigere load + filter (kan testes efter baseline forbedringer).
Foreslået redesign (ingen kode ændret endnu)
Nye komponenter / trin:

Index-generator (kan integreres i make_rebased_windows.py eller separat script):

Output: rebased_index.parquet
Kolonner: Ticker, RefDate, SourceFile, per match-col f.eks. AdjClose_Rebased_sig (liste / fast-længde array), evt. LenValid
Signatur: Downsample vinduesvektor til d punkter (f.eks. d=8–16) med lineær sampling eller PAA (Piecewise Aggregate Approximation).
CLI-udvidelser i analog_matcher_watch.py:

--use-index (bool): Aktiver index-baseret prefilter.
--index-file rebased_index.parquet
--sig-downsample 8
--prefilter-topk 500
--score-workers N (antal parallelle workers til fuld scoring; default 1)
--cache-ref (behold allerede læste referencefiler i memory mellem tickere)
--list-only (print antal vinduer / index stats og afslut – nyttigt til tuning)
(Evt.) --metric corr|mse (findes allerede – vi bevarer)
Arkitekturændring:

Adskil “query build” og “candidate retrieval” og “scoring”.
Introducer et internt objekt ReferenceStore:
Metoder: load_index(), prefilter(query_sig) -> candidate rows, load_windows(grouped by SourceFile).
Fil-cache: dict: filename -> DataFrame (weakref eller LRU med memory-limit).
Scoringfase:
Vectoriser: Stak alle kandidaters match_cols arrays til shape (K, lookback) og beregn MSE i én NumPy operation:
diff = candidates - query (broadcast), mse = (diff**2).mean(axis=2)
Weighted: vægt pr. col før sum.
Corr: Brug normaliseret dot product:
Standardiser query og candidates én gang.
Parallelisering (fase 2 – efter baseline):

Hvis K stadig stor, chunk kandidater i segmenter og process pool (multiprocessing) – især for corr som er dyrere.
Yderligere forbedringer (fase 3):

Numba JIT på scoring (kan give 2–5x afh. på CPU).
Persistér signaturer også normaliseret (for corr) for at spare beregning.
Bloom filter / locality sensitive hashing (LSH) hvis N bliver meget stort.
Minim