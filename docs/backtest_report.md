# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-04-04  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.525 | 48,810 | 0.460 | 1,346 | no |
| 0.45 | 0.550 | 36,536 | 0.516 | 577 | no |
| 0.50 | 0.580 | 24,329 | 0.619 | 202 | no |
| 0.55 | 0.607 | 14,393 | 0.672 | 61 | no <-- 60% flag <-- OOS breakeven |
| 0.60 | 0.632 | 7,074 | (0.667, <20t) | 9 | no |
| 0.65 | 0.657 | 2,832 | 0 trades | 0 | no |
| 0.70 | 0.655 | 857 | 0 trades | 0 | no |
| 0.75 | 0.623 | 191 | 0 trades | 0 | no |
| 0.80 | (0.588, <20t) | 17 | 0 trades | 0 | no |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.55 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.55)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.55 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| ORCL | tech | 1118 | 69.2% | +0.115% | +11.54% | +3.45% |
| TSLA | tech | 0 | N/A | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| INTC | tech | 586 | 63.5% | -0.143% | -14.33% | -10.98% |
| META | tech | 428 | 62.8% | -0.172% | -17.17% | +48.93% |
| PFE | biotech | 1726 | 62.6% | -0.182% | -18.16% | +6.68% |
| MSFT | tech | 2054 | 62.4% | -0.191% | -19.13% | +21.39% |
| LLY | biotech | 1545 | 61.8% | -0.221% | -22.14% | +16.90% |
| BMY | biotech | 1839 | 60.2% | -0.289% | -28.87% | +25.10% |
| ADBE | tech | 910 | 59.9% | -0.305% | -30.49% | +30.47% |
| GOOGL | tech | 926 | 59.2% | -0.337% | -33.69% | +34.00% |
| ABBV | biotech | 263 | 58.9% | -0.348% | -34.79% | +52.80% |
| AMZN | tech | 383 | 58.8% | -0.356% | -35.64% | +24.03% |
| CRM | tech | 333 | 58.3% | -0.378% | -37.84% | -3.38% |
| AMGN | biotech | 1176 | 57.6% | -0.409% | -40.94% | +11.85% |
| BIIB | biotech | 299 | 51.5% | -0.682% | -68.23% | +71.51% |
| NVDA | tech | 14 | 50.0% | -0.750% | -75.00% | +86.15% |
| AAPL | tech | 415 | 49.9% | -0.755% | -75.54% | +88.96% |
| GILD | biotech | 326 | 49.4% | -0.778% | -77.76% | +60.03% |
| REGN | biotech | 11 | 45.5% | -0.955% | -95.45% | -1.14% |
| VRTX | biotech | 36 | 36.1% | -1.375% | -137.50% | +7.27% |
| AMD | tech | 5 | 0.0% | -3.000% | -300.00% | +164.48% |

---

## P&L at OOS-Breakeven Threshold (0.55)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.55 is ~67.2% < 66.7%.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| ORCL | tech | 1118 | 69.2% | +0.115% | +11.54% | +3.45% |
| TSLA | tech | 0 | N/A | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| INTC | tech | 586 | 63.5% | -0.143% | -14.33% | -10.98% |
| META | tech | 428 | 62.8% | -0.172% | -17.17% | +48.93% |
| PFE | biotech | 1726 | 62.6% | -0.182% | -18.16% | +6.68% |
| MSFT | tech | 2054 | 62.4% | -0.191% | -19.13% | +21.39% |
| LLY | biotech | 1545 | 61.8% | -0.221% | -22.14% | +16.90% |
| BMY | biotech | 1839 | 60.2% | -0.289% | -28.87% | +25.10% |
| ADBE | tech | 910 | 59.9% | -0.305% | -30.49% | +30.47% |
| GOOGL | tech | 926 | 59.2% | -0.337% | -33.69% | +34.00% |
| ABBV | biotech | 263 | 58.9% | -0.348% | -34.79% | +52.80% |
| AMZN | tech | 383 | 58.8% | -0.356% | -35.64% | +24.03% |
| CRM | tech | 333 | 58.3% | -0.378% | -37.84% | -3.38% |
| AMGN | biotech | 1176 | 57.6% | -0.409% | -40.94% | +11.85% |
| BIIB | biotech | 299 | 51.5% | -0.682% | -68.23% | +71.51% |
| NVDA | tech | 14 | 50.0% | -0.750% | -75.00% | +86.15% |
| AAPL | tech | 415 | 49.9% | -0.755% | -75.54% | +88.96% |
| GILD | biotech | 326 | 49.4% | -0.778% | -77.76% | +60.03% |
| REGN | biotech | 11 | 45.5% | -0.955% | -95.45% | -1.14% |
| VRTX | biotech | 36 | 36.1% | -1.375% | -137.50% | +7.27% |
| AMD | tech | 5 | 0.0% | -3.000% | -300.00% | +164.48% |

### Holdout (2024+)

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |
|--------|--------|--------|:--------:|:----------:|:----------:|
| MSFT | tech | 52 | 73.1% | +0.288% | +28.85% |
| NVDA | tech | 0 | N/A | N/A | N/A |
| GOOGL | tech | 0 | N/A | N/A | N/A |
| AMZN | tech | 0 | N/A | N/A | N/A |
| AMD | tech | 0 | N/A | N/A | N/A |
| TSLA | tech | 0 | N/A | N/A | N/A |
| CRM | tech | 0 | N/A | N/A | N/A |
| ADBE | tech | 0 | N/A | N/A | N/A |
| INTC | tech | 0 | N/A | N/A | N/A |
| ORCL | tech | 0 | N/A | N/A | N/A |
| LLY | biotech | 0 | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A |
| BIIB | biotech | 0 | N/A | N/A | N/A |
| REGN | biotech | 0 | N/A | N/A | N/A |
| VRTX | biotech | 0 | N/A | N/A | N/A |
| ABBV | biotech | 0 | N/A | N/A | N/A |
| BMY | biotech | 0 | N/A | N/A | N/A |
| GILD | biotech | 0 | N/A | N/A | N/A |
| AMGN | biotech | 0 | N/A | N/A | N/A |
| PFE | biotech | 0 | N/A | N/A | N/A |
| AAPL | tech | 7 | 42.9% | -1.071% | -107.14% |
| META | tech | 2 | 0.0% | -3.000% | -300.00% |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 7,172 | 61.7% | -0.225% | -22.48% | no |
| biotech | 7,221 | 59.7% | -0.315% | -31.53% | no |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.55, 14,393 trades, win rate 60.7%, avg return -0.270%/trade (-27.02% per 100 trades).  OOS win rate below 66.7% breakeven -- strategy not yet profitable even on OOS.

**Holdout (2025+, clean):** 61 trades, win rate 67.2%, avg return +0.025%/trade (+2.46% per 100 trades).  **Holdout breakeven exceeded.**

**Root cause of OOS/holdout gap:**  
OOS precision (~71%) reflects historical Sideways periods the model was
implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,
more volatile window (post-rate-hike cycle, AI bull market) where the
market moved directionally more often than the model expected.
With only ~304 holdout rows per ticker (~1.2 years), confidence intervals
are wide. Re-evaluate once 2025-2026 data accumulates.

**Best current candidate for cautious paper trading:**  
- **MSFT** (tech): holdout win rate 73.1% (52 trades), avg +0.288%/trade

---

*Generated by src/pipeline/10_backtest.py*
