from pathlib import Path
import pandas as pd
p=Path('data/flat_vectors/all_vectors.parquet')
if not p.exists():
    p=Path('data/flat_vectors/all_vectors.csv')
print('Using',p)
if p.suffix=='.parquet':
    df=pd.read_parquet(p)
else:
    df=pd.read_csv(p)
print('shape',df.shape)
print('cols count',len(df.columns))
print(df.columns.tolist()[:120])
print('sample rows:')
print(df.head(3).to_string(index=False))
