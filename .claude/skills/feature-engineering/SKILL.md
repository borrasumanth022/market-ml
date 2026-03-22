# Skill: feature-engineering

How we build technical features for any ticker in market_ml.

## The 36-feature set (inherited from aapl_ml Phase 2)

These features are the result of aapl_ml's Phase 2 feature selection process.
They are the inputs to every ML model in this project. **Do not add features to
this list without running the correlation check first.**

### Feature groups

**A. Price & volume transforms (6 features)**
| Feature | How computed | Why kept |
|---------|-------------|---------|
| `return_1d` | `close.pct_change(1)` | Raw daily momentum |
| `return_2d` | `close.pct_change(2)` | 2-day momentum |
| `return_5d` | `close.pct_change(5)` | Weekly momentum |
| `volume_zscore` | `(vol - roll20_mean) / roll20_std` | Volume anomaly detection |
| `price_52w_pct` | `(close - 52w_low) / (52w_high - 52w_low)` | Position in yearly range |
| `gap_pct` | `(open - prev_close) / prev_close` | Overnight gap signal |

**B. Trend indicators (8 features)**
| Feature | How computed | Why kept |
|---------|-------------|---------|
| `close_vs_sma10` | `close / sma10 - 1` | Short-term trend deviation |
| `close_vs_sma20` | `close / sma20 - 1` | Medium-short trend |
| `close_vs_sma50` | `close / sma50 - 1` | Medium trend |
| `close_vs_sma100` | `close / sma100 - 1` | Medium-long trend |
| `close_vs_sma200` | `close / sma200 - 1` | Long-term trend (top SHAP feature) |
| `cross_50_200` | `(sma50 > sma200).astype(int)` | Golden/death cross flag |
| `macd_hist` | `(ema12 - ema26) - macd_signal` | Trend momentum divergence |
| `macd_signal` | `EMA9(macd)` | MACD smoothed signal |

**C. Momentum indicators (4 features)**
| Feature | How computed | Why kept |
|---------|-------------|---------|
| `rsi_14` | Standard RSI, 14-day | Overbought/oversold |
| `rsi_7` | Standard RSI, 7-day | Shorter-term momentum |
| `roc_10` | `close.pct_change(10) * 100` | 10-day rate of change |
| `roc_21` | `close.pct_change(21) * 100` | Monthly rate of change |
| `stoch_k` | `100 * (close - lo14) / (hi14 - lo14)` | Stochastic oscillator |
| `stoch_d` | `stoch_k.rolling(3).mean()` | Stochastic signal line |

**D. Volatility indicators (5 features)**
| Feature | How computed | Why kept |
|---------|-------------|---------|
| `atr_pct` | `atr14 / close` | Top SHAP feature overall |
| `bb_width` | `(bb_upper - bb_lower) / sma20` | Volatility expansion/contraction |
| `bb_pct` | `(close - bb_lower) / (bb_upper - bb_lower)` | Position within bands |
| `hvol_10d` | `log_returns.rolling(10).std() * sqrt(252)` | Short-term realised vol |
| `hvol_21d` | `log_returns.rolling(21).std() * sqrt(252)` | Monthly vol (top Bear driver) |
| `hvol_63d` | `log_returns.rolling(63).std() * sqrt(252)` | Quarterly vol (top Bull driver) |

**E. Microstructure / candle features (5 features)**
| Feature | How computed | Why kept |
|---------|-------------|---------|
| `candle_body` | `abs(close - open) / (high - low)` | Body-to-range ratio |
| `candle_dir` | `sign(close - open)` | Up or down day |
| `upper_shadow` | `(high - max(open,close)) / (high - low)` | Upper wick size |
| `lower_shadow` | `(min(open,close) - low) / (high - low)` | Lower wick size |
| `hl_range_pct` | `(high - low) / close` | Intraday range |

**F. Calendar features (6 features)**
| Feature | Notes |
|---------|-------|
| `day_of_week` | 0=Mon, 4=Fri |
| `month` | 1–12 |
| `is_month_end` | Boolean flag |
| `is_month_start` | Boolean flag |
| `is_quarter_end` | Boolean flag |

### Features deliberately excluded

**Dropped as price-level proxies** (absolute values, not normalised):
`log_close, sma_10/20/50/100/200, ema_12, ema_26, bb_upper, bb_lower, atr_14, macd`

**Dropped by correlation (|r| > 0.95)**:
`log_return_1d` (≈ return_1d), `quarter` (≈ month), `roc_5` (≈ return_5d), `williams_r` (≈ stoch_k)

---

## Running feature engineering

```bash
# All tickers
C:\Users\borra\anaconda3\python.exe src\pipeline\02_features.py

# Single ticker
C:\Users\borra\anaconda3\python.exe src\pipeline\02_features.py AAPL
```

## Output
- `data/processed/{TICKER}_features.parquet`
- Contains all raw OHLCV columns **plus** all computed feature columns (57 total before
  feature selection). The selected 36 are a subset used during model training.
- Warm-up rows dropped: first 200 rows lost to the 200-day SMA. For short-history tickers
  (MRNA: 1,830 rows), this removes ~11% of data — acceptable.

## Adding a new feature group
1. Write a new `add_<group>(df) -> df` function following the same pattern
2. Call it in `build_features()` after the existing steps
3. Run feature selection again (`src/pipeline/04_feature_selection.py` — TODO) to check
   for correlations before promoting any new feature to the 36-feature model set
4. **Never** compute a feature using future data. All rolling windows look backward only.

## Warm-up / NaN handling
Rolling windows create NaN rows at the start of each ticker's history:
- 200-day SMA needs 200 rows → first 199 rows are dropped
- This is done via `df.dropna(subset=["sma_200"])` at the end of `build_features()`
- All other features that need fewer than 200 days will also be clean after this drop

Do not fill NaN values with 0 or forward-fill in the feature engineering step. Let them
propagate naturally; the dropna on sma_200 catches the vast majority.
