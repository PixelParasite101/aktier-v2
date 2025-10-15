#!/usr/bin/env python3
"""Plot similarity matches for watch ticker.

Reads:
  data/flat_vectors/watch_query_vector.csv  (produced by compare_watch_with_flat_vectors.py)
  data/flat_vectors/watch_matches_full.csv  (full ranked matches with sim_to_watch)

Creates plots (PNG) under data/flat_vectors/plots/ by default:
  - similarity_bar_topN.png  (bar chart of cosine similarity for top N)
  - heatmaps_window_features.png  (grid: query + each candidate window as heatmap)
  - feature_lines_<ticker>.png (for each candidate: line comparison per feature vs query)

CLI:
  --top N (default 5)
  --out-dir DIR (default: data/flat_vectors/plots)
  --no-show (suppress interactive show)
  --style STYLE (matplotlib style, default 'seaborn-v0_8' if available else 'default')
  --dpi DPI (default 120)
  --features-labels comma,separated,list (override generic f0,f1,... labels)
  --limit-lines N (limit how many candidates to produce per-feature line plots for; default = top N)

Assumptions:
  - Query vector file has columns: Ticker, WindowUsed, FeaturesUsed, VectorLength, v0..vK
  - Matches file has columns including sim_to_watch and v0..vK
  - All candidate rows share same flattened length as query (or extra columns ignored)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import math

import warnings

try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("Fejl: matplotlib er ikke installeret. Installer med: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e

# Optional seaborn for nicer heatmaps
try:  # pragma: no cover
    import seaborn as sns
    _HAVE_SNS = True
except Exception:  # pragma: no cover
    _HAVE_SNS = False

DEFAULT_QUERY_CSV = Path('data/flat_vectors/watch_query_vector.csv')
DEFAULT_MATCHES_CSV = Path('data/flat_vectors/watch_matches_full.csv')


def parse_args():
    p = argparse.ArgumentParser(description='Plot watch similarity matches')
    p.add_argument('--query-csv', default=str(DEFAULT_QUERY_CSV), help='Path til query vector CSV')
    p.add_argument('--matches-csv', default=str(DEFAULT_MATCHES_CSV), help='Path til fuld matches CSV (med sim_to_watch)')
    p.add_argument('--top', type=int, default=5, help='Antal top matches at plotte (default 5)')
    p.add_argument('--out-dir', default='data/flat_vectors/plots', help='Output mappe til plots')
    p.add_argument('--no-show', action='store_true', help='Ingen plt.show() (kun gem)')
    p.add_argument('--style', default=None, help='Matplotlib style (default seaborn-v0_8 hvis tilgængelig)')
    p.add_argument('--dpi', type=int, default=120, help='Figur DPI (default 120)')
    p.add_argument('--features-labels', default=None, help='Komma-sep labels for funktioner (ellers f0,f1,...)')
    p.add_argument('--limit-lines', type=int, default=None, help='Maks antal kandidater der får feature line-plots (default = top)')
    return p.parse_args()


def _apply_style(style: str | None):
    if style:
        try:
            plt.style.use(style)
            return
        except Exception:
            print(f"Advarsel: style '{style}' ikke fundet. Bruger default.")
    else:
        # fallback chain
        for s in ('seaborn-v0_8', 'seaborn', 'default'):
            try:
                plt.style.use(s)
                return
            except Exception:
                continue


def load_query(path: Path) -> tuple[pd.Series, np.ndarray, int, int]:
    if not path.exists():
        raise SystemExit(f'Query CSV ikke fundet: {path}')
    q_df = pd.read_csv(path)
    if q_df.empty:
        raise SystemExit('Query CSV er tom')
    row = q_df.iloc[0]
    vector_cols = [c for c in q_df.columns if c.startswith('v')]
    v = row[vector_cols].to_numpy(dtype=float)
    window = int(row.get('WindowUsed', math.nan)) if 'WindowUsed' in row else None
    n_features = int(row.get('FeaturesUsed', math.nan)) if 'FeaturesUsed' in row else None
    if window and n_features and window * n_features != len(vector_cols):
        print(f"Advarsel: metadata (window*features)={window*n_features} matcher ikke antal v-kolonner={len(vector_cols)}. Ignorerer metadata.")
        window = None
        n_features = None
    return row, v, window, n_features


def load_matches(path: Path, top: int, target_len: int) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f'Matches CSV ikke fundet: {path}')
    df = pd.read_csv(path)
    if 'sim_to_watch' not in df.columns:
        raise SystemExit('Kolonne sim_to_watch mangler i matches CSV')
    df = df.sort_values('sim_to_watch', ascending=False).head(top)
    # filter vector columns to target_len
    vec_cols = [c for c in df.columns if c.startswith('v')]
    if len(vec_cols) < target_len:
        raise SystemExit(f'Mangler v-kolonner: har {len(vec_cols)}, forventede mindst {target_len}')
    use_cols = vec_cols[:target_len]
    return df, use_cols


def reconstruct_matrix(flat_vec: np.ndarray, window: int | None, n_features: int | None) -> np.ndarray:
    if window is None or n_features is None:
        # heuristik: prøv at faktoriser længden
        L = len(flat_vec)
        # prøv alle divisorer <= 200
        for w in range(5, 301):
            if L % w == 0:
                f = L // w
                if 1 <= f <= 64:  # rimelig feature range
                    return flat_vec.reshape(w, f)
        # fallback: 1 x L
        return flat_vec.reshape(1, -1)
    try:
        return flat_vec.reshape(window, n_features)
    except Exception:
        return flat_vec.reshape(1, -1)


def plot_similarity_bars(matches: pd.DataFrame, out_dir: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(matches['Ticker'].astype(str) + '\n' + matches['RefDate'].astype(str), matches['sim_to_watch'], color='tab:blue')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Top matches')
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(matches['sim_to_watch']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    out_path = out_dir / 'similarity_bar_topN.png'
    fig.savefig(out_path, dpi=dpi)
    return out_path


def plot_heatmaps(query_mat: np.ndarray, candidate_mats: list[tuple[str, np.ndarray]], out_dir: Path, feature_labels: list[str], dpi: int):
    # Grid: 1 + N heatmaps
    n = 1 + len(candidate_mats)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.6 * rows), squeeze=False)
    def _do(ax, mat, title):
        if _HAVE_SNS:
            sns.heatmap(mat, ax=ax, cmap='viridis', cbar=False)
        else:
            im = ax.imshow(mat, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Feature')
        ax.set_ylabel('Day')
    _do(axes[0][0], query_mat, 'Query')
    for idx, (label, mat) in enumerate(candidate_mats, start=1):
        r = idx // cols
        c = idx % cols
        _do(axes[r][c], mat, label)
    # Remove empty axes
    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r][c].axis('off')
    fig.suptitle('Window x Feature heatmaps', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = out_dir / 'heatmaps_window_features.png'
    fig.savefig(out_path, dpi=dpi)
    return out_path


def plot_feature_lines(query_mat: np.ndarray, candidate_mats: list[tuple[str, np.ndarray]], out_dir: Path, feature_labels: list[str], limit: int | None, dpi: int):
    limit = limit or len(candidate_mats)
    limit = min(limit, len(candidate_mats))
    W, F = query_mat.shape
    for idx, (label, mat) in enumerate(candidate_mats[:limit]):
        fig, axes = plt.subplots(math.ceil(F / 3), 3, figsize=(10, 2.4 * math.ceil(F / 3)), squeeze=False)
        for f in range(F):
            r = f // 3
            c = f % 3
            ax = axes[r][c]
            ax.plot(range(W), query_mat[:, f], label='Query', linewidth=1.5)
            if mat.shape == query_mat.shape:
                ax.plot(range(W), mat[:, f], label=label, linewidth=1.0)
            else:
                # length mismatch -> skip candidate line
                pass
            ax.set_title(feature_labels[f] if f < len(feature_labels) else f'f{f}', fontsize=8)
            ax.tick_params(labelsize=7)
        # turn off unused axes
        total_axes = axes.size
        for k in range(F, total_axes):
            rr = k // 3
            cc = k % 3
            axes[rr][cc].axis('off')
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper right')
        fig.suptitle(f'Feature lines vs Query - {label}', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        # Sanitér label til filnavn: fjern newline, kolon, mellemrum mv.
        safe_label = label.replace('\n', '_').replace('/', '_').replace('\\', '_').replace(':', '-')
        safe_label = ''.join(ch for ch in safe_label if ch.isalnum() or ch in '._-')
        out_path = out_dir / f'feature_lines_{safe_label}.png'
        fig.savefig(out_path, dpi=dpi)
    return True


def main():  # pragma: no cover (script usage)
    args = parse_args()
    _apply_style(args.style)

    query_row, q_vec, q_window, q_feats = load_query(Path(args.query_csv))
    vec_len = len(q_vec)
    matches_df, vec_cols = load_matches(Path(args.matches_csv), args.top, vec_len)

    # Reconstruct matrices
    if q_window and q_feats and q_window * q_feats == vec_len:
        query_mat = q_vec.reshape(q_window, q_feats)
    else:
        query_mat = reconstruct_matrix(q_vec, q_window, q_feats)
        q_window, q_feats = query_mat.shape

    candidate_mats: list[tuple[str, np.ndarray]] = []
    for _, row in matches_df.iterrows():
        flat = row[vec_cols].to_numpy(dtype=float)
        try:
            mat = flat.reshape(q_window, q_feats)
        except Exception:
            mat = reconstruct_matrix(flat, q_window, q_feats)
        label = f"{row['Ticker']}\n{row['RefDate']}\n{row['sim_to_watch']:.3f}"
        candidate_mats.append((label, mat))

    # Feature labels
    if args.features_labels:
        feature_labels = [s.strip() for s in args.features_labels.split(',') if s.strip()]
    else:
        feature_labels = [f'f{i}' for i in range(q_feats)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bar_path = plot_similarity_bars(matches_df, out_dir, args.dpi)
    heat_path = plot_heatmaps(query_mat, candidate_mats, out_dir, feature_labels, args.dpi)
    plot_feature_lines(query_mat, candidate_mats, out_dir, feature_labels, args.limit_lines, args.dpi)

    print('Gemte plots:')
    print('  ', bar_path)
    print('  ', heat_path)
    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == '__main__':
    main()
