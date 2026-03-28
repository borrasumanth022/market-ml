# Python Coding Standards — market_ml

## File paths — always pathlib, never raw strings
```python
# Good
from pathlib import Path
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
path = DATA_DIR / f"{ticker}_features.parquet"

# Bad — hardcoded absolute path
path = r"C:\Users\...\data\processed\AAPL_features.parquet"
```

## Parquet I/O
Always use `engine="pyarrow"`. Index must be a `DatetimeIndex` named `date`.

```python
df = pd.read_parquet(path, engine="pyarrow")
df.to_parquet(path, engine="pyarrow", index=True)
assert df.index.name == "date"
assert pd.api.types.is_datetime64_any_dtype(df.index)
```

## Skip-if-exists — every pipeline script must implement this
```python
if output_path.exists():
    print(f"[SKIP] {output_path.name} already exists. Delete to re-run.")
    sys.exit(0)
```

## Fail loudly — never swallow exceptions silently
```python
# Good
if df.empty:
    raise ValueError(f"Empty DataFrame from {path}. Expected >{min_rows} rows.")

# Bad
if df.empty:
    return  # silent failure masks data problems
```

## ASCII-only print output
Windows cp1252 terminals crash on Unicode. Use only ASCII in all print() calls.
```python
# Good
print(f"[OK]   {ticker}: {len(df)} rows, {df.shape[1]} cols")
print(f"[FAIL] {ticker}: missing column 'close'")

# Bad — will crash on Windows cp1252
print(f"✓ {ticker}: {len(df)} rows")
print(f"→ Processing {ticker}")
```
Use `[OK]`, `[SKIP]`, `[WARN]`, `[FAIL]` prefixes for pipeline output.

## Ticker registry — always import from config
```python
# Good
from config.tickers import SECTORS, ALL_TICKERS

# Bad — hardcoded lists diverge from the source of truth
TICKERS = ["AAPL", "MSFT", "NVDA"]
```

## DataFrame column naming
- Lowercase snake_case: `return_1d`, `hvol_21d`, `close_vs_sma50`
- No spaces, no camelCase, no uppercase
- Prefix by category: `rsi_`, `macd_`, `bb_`, `hvol_`, `sma_`, `ema_`

## Type hints — required on all function signatures
```python
def compute_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ...
```

## Imports order
1. Standard library (`sys`, `pathlib`, `datetime`)
2. Third-party (`pandas`, `numpy`, `sklearn`, `xgboost`, `shap`)
3. Local (`from config.tickers import ...`)
Separated by blank lines. No `import *`.
