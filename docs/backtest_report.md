# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-03-29  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.496 | 23,280 | 0.449 | 740 | no |
| 0.45 | 0.519 | 17,272 | 0.506 | 344 | no |
| 0.50 | 0.544 | 11,383 | 0.606 | 127 | no |
| 0.55 | 0.573 | 6,896 | 0.690 | 42 | no <-- OOS breakeven |
| 0.60 | 0.594 | 3,758 | (0.778, <20t) | 9 | no |
| 0.65 | 0.604 | 1,791 | (1.000, <20t) | 1 | no <-- 60% flag |
| 0.70 | 0.640 | 702 | 0 trades | 0 | no |
| 0.75 | 0.655 | 174 | 0 trades | 0 | no |
| 0.80 | 0.469 | 32 | 0 trades | 0 | no |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.55 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.65)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.65 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| NVDA | tech | 4 | 75.0% | +0.375% | +37.50% | -87.90% |
| MSFT | tech | 717 | 67.1% | +0.019% | +1.88% | -6.66% |
| META | tech | 9 | 66.7% | +0.000% | +0.00% | -26.14% |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| LLY | biotech | 597 | 60.1% | -0.294% | -29.40% | +46.65% |
| BIIB | biotech | 136 | 56.6% | -0.452% | -45.22% | +66.20% |
| AMZN | tech | 26 | 53.8% | -0.577% | -57.69% | -43.59% |
| VRTX | biotech | 17 | 52.9% | -0.618% | -61.76% | +108.06% |
| REGN | biotech | 18 | 50.0% | -0.750% | -75.00% | +83.19% |
| GOOGL | tech | 178 | 49.4% | -0.775% | -77.53% | +136.71% |
| AAPL | tech | 89 | 39.3% | -1.230% | -123.03% | +33.54% |

---

## P&L at OOS-Breakeven Threshold (0.55)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.55 is ~69.0% < 66.7%.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| META | tech | 142 | 64.1% | -0.116% | -11.62% | -9.04% |
| MSFT | tech | 1832 | 62.2% | -0.202% | -20.22% | +23.22% |
| LLY | biotech | 1999 | 60.1% | -0.294% | -29.41% | +32.38% |
| NVDA | tech | 30 | 60.0% | -0.300% | -30.00% | +41.27% |
| GOOGL | tech | 937 | 59.3% | -0.330% | -32.98% | +51.32% |
| AMZN | tech | 341 | 51.9% | -0.664% | -66.42% | +18.74% |
| BIIB | biotech | 670 | 51.0% | -0.703% | -70.30% | +82.21% |
| AAPL | tech | 513 | 45.4% | -0.956% | -95.61% | +72.64% |
| REGN | biotech | 173 | 45.1% | -0.971% | -97.11% | +130.18% |
| VRTX | biotech | 259 | 43.2% | -1.054% | -105.41% | -0.35% |

### Holdout (2024+)

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |
|--------|--------|--------|:--------:|:----------:|:----------:|
| MSFT | tech | 27 | 77.8% | +0.500% | +50.00% |
| NVDA | tech | 0 | N/A | N/A | N/A |
| GOOGL | tech | 0 | N/A | N/A | N/A |
| META | tech | 0 | N/A | N/A | N/A |
| LLY | biotech | 0 | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A |
| BIIB | biotech | 0 | N/A | N/A | N/A |
| REGN | biotech | 0 | N/A | N/A | N/A |
| VRTX | biotech | 0 | N/A | N/A | N/A |
| AAPL | tech | 14 | 57.1% | -0.429% | -42.86% |
| AMZN | tech | 1 | 0.0% | -3.000% | -300.00% |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 3,795 | 58.3% | -0.375% | -37.47% | no |
| biotech | 3,101 | 55.9% | -0.484% | -48.37% | no |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.55, 6,896 trades, win rate 57.2%, avg return -0.424%/trade (-42.37% per 100 trades).  OOS win rate below 66.7% breakeven -- strategy not yet profitable even on OOS.

**Holdout (2025+, clean):** 42 trades, win rate 69.0%, avg return +0.107%/trade (+10.71% per 100 trades).  **Holdout breakeven exceeded.**

**Root cause of OOS/holdout gap:**  
OOS precision (~71%) reflects historical Sideways periods the model was
implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,
more volatile window (post-rate-hike cycle, AI bull market) where the
market moved directionally more often than the model expected.
With only ~304 holdout rows per ticker (~1.2 years), confidence intervals
are wide. Re-evaluate once 2025-2026 data accumulates.

**Best current candidate for cautious paper trading:**  
- **MSFT** (tech): holdout win rate 77.8% (27 trades), avg +0.500%/trade

---

*Generated by src/pipeline/10_backtest.py*
