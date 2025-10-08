"""
Diagnose script for flat_vectorize input data.
Prints overall rows, number of groups, max/min/median group sizes and how many groups meet a given window size.
"""
from pathlib import Path
import pandas as pd
import sys

p_csv = Path('data') / 'rebased' / 'rebased_all.csv'
p_pq = Path('data') / 'rebased' / 'rebased_all.parquet'
if p_pq.exists():
    df = pd.read_parquet(p_pq)
elif p_csv.exists():
    df = pd.read_csv(p_csv)
else:
    print('No input data found at data/rebased/rebased_all.(csv|parquet)')
    sys.exit(1)

print('Total rows:', len(df))
if 'Ticker' not in df.columns or 'RefDate' not in df.columns:
    print('Missing Ticker or RefDate columns')
    sys.exit(1)

groups = df.groupby(['Ticker', 'RefDate'])
sizes = groups.size().reset_index(name='n')
print('Number of (Ticker,RefDate) groups:', len(sizes))
print('Group sizes: min=%d, median=%d, mean=%.2f, max=%d' % (sizes['n'].min(), sizes['n'].median(), sizes['n'].mean(), sizes['n'].max()))

for w in [75, 50, 25, 10, 5]:
    meet = (sizes['n'] >= w).sum()
    print(f'Groups with >={w} rows: {meet} ({meet/len(sizes):.2%})')

# show a few example groups with small sizes
small = sizes.nsmallest(5, 'n')
print('\n5 smallest groups:')
print(small.to_string(index=False))

# show the biggest groups
big = sizes.nlargest(5, 'n')
print('\n5 largest groups:')
print(big.to_string(index=False))
