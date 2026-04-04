# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-04-04  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.536 | 59,715 | 0.461 | 2,012 | no |
| 0.45 | 0.561 | 45,048 | 0.504 | 936 | no |
| 0.50 | 0.591 | 30,548 | 0.587 | 341 | no |
| 0.55 | 0.620 | 18,765 | 0.654 | 104 | no <-- 60% flag |
| 0.60 | 0.649 | 10,001 | 0.840 | 25 | no |
| 0.65 | 0.677 | 4,657 | (1.000, <20t) | 6 | YES <-- OOS breakeven |
| 0.70 | 0.693 | 1,875 | 0 trades | 0 | YES |
| 0.75 | 0.723 | 672 | 0 trades | 0 | YES |
| 0.80 | 0.754 | 187 | 0 trades | 0 | YES |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.65 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.55)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.55 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| WFC | financials | 1627 | 70.2% | +0.161% | +16.13% | +10.66% |
| ORCL | tech | 1118 | 69.2% | +0.115% | +11.54% | +3.45% |
| TSLA | tech | 0 | N/A | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| JPM | financials | 1060 | 66.2% | -0.020% | -1.98% | +41.67% |
| BAC | financials | 761 | 65.8% | -0.037% | -3.75% | +7.68% |
| INTC | tech | 586 | 63.5% | -0.143% | -14.33% | -10.98% |
| META | tech | 428 | 62.8% | -0.172% | -17.17% | +48.93% |
| PFE | biotech | 1726 | 62.6% | -0.182% | -18.16% | +6.68% |
| MSFT | tech | 2054 | 62.4% | -0.191% | -19.13% | +21.39% |
| LLY | biotech | 1545 | 61.8% | -0.221% | -22.14% | +16.90% |
| MS | financials | 354 | 61.0% | -0.254% | -25.42% | +21.74% |
| BMY | biotech | 1839 | 60.2% | -0.289% | -28.87% | +25.10% |
| ADBE | tech | 910 | 59.9% | -0.305% | -30.49% | +30.47% |
| GOOGL | tech | 926 | 59.2% | -0.337% | -33.69% | +34.00% |
| ABBV | biotech | 263 | 58.9% | -0.348% | -34.79% | +52.80% |
| AMZN | tech | 383 | 58.8% | -0.356% | -35.64% | +24.03% |
| GS | financials | 570 | 58.4% | -0.371% | -37.11% | +42.02% |
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

## P&L at OOS-Breakeven Threshold (0.65)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.65 is ~100.0% < 66.7%.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| META | tech | 169 | 76.9% | +0.462% | +46.15% | +43.18% |
| INTC | tech | 37 | 75.7% | +0.405% | +40.54% | +13.90% |
| WFC | financials | 953 | 73.1% | +0.291% | +29.12% | +6.31% |
| GOOGL | tech | 117 | 71.8% | +0.231% | +23.08% | +46.58% |
| JPM | financials | 421 | 70.5% | +0.175% | +17.46% | +35.17% |
| ADBE | tech | 213 | 69.5% | +0.127% | +12.68% | +41.18% |
| MSFT | tech | 746 | 69.4% | +0.125% | +12.47% | +4.04% |
| MS | financials | 52 | 69.2% | +0.115% | +11.54% | +12.43% |
| PFE | biotech | 302 | 67.9% | +0.055% | +5.46% | -8.68% |
| BAC | financials | 288 | 67.4% | +0.031% | +3.12% | +9.65% |
| NVDA | tech | 0 | N/A | N/A | N/A | N/A |
| AMD | tech | 0 | N/A | N/A | N/A | N/A |
| TSLA | tech | 0 | N/A | N/A | N/A | N/A |
| CRM | tech | 84 | 66.7% | +0.000% | +0.00% | -73.17% |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| REGN | biotech | 0 | N/A | N/A | N/A | N/A |
| ABBV | biotech | 3 | 66.7% | +0.000% | +0.00% | -155.26% |
| ORCL | tech | 171 | 64.9% | -0.079% | -7.89% | -21.91% |
| GS | financials | 111 | 63.1% | -0.162% | -16.22% | +53.51% |
| BMY | biotech | 406 | 61.1% | -0.251% | -25.12% | +2.88% |
| AMZN | tech | 51 | 60.8% | -0.265% | -26.47% | +57.81% |
| LLY | biotech | 289 | 57.8% | -0.400% | -39.97% | +15.31% |
| AMGN | biotech | 164 | 56.7% | -0.448% | -44.82% | +3.23% |
| GILD | biotech | 34 | 55.9% | -0.485% | -48.53% | +50.70% |
| AAPL | tech | 21 | 47.6% | -0.857% | -85.71% | -6.78% |
| BIIB | biotech | 20 | 40.0% | -1.200% | -120.00% | +37.10% |
| VRTX | biotech | 5 | 40.0% | -1.200% | -120.00% | +290.59% |

### Holdout (2024+)

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |
|--------|--------|--------|:--------:|:----------:|:----------:|
| JPM | financials | 2 | 100.0% | +1.500% | +150.00% |
| BAC | financials | 2 | 100.0% | +1.500% | +150.00% |
| WFC | financials | 2 | 100.0% | +1.500% | +150.00% |
| AAPL | tech | 0 | N/A | N/A | N/A |
| MSFT | tech | 0 | N/A | N/A | N/A |
| NVDA | tech | 0 | N/A | N/A | N/A |
| GOOGL | tech | 0 | N/A | N/A | N/A |
| AMZN | tech | 0 | N/A | N/A | N/A |
| META | tech | 0 | N/A | N/A | N/A |
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
| GS | financials | 0 | N/A | N/A | N/A |
| MS | financials | 0 | N/A | N/A | N/A |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 1,609 | 69.4% | +0.121% | +12.12% | YES |
| biotech | 1,223 | 60.8% | -0.263% | -26.25% | no |
| financials | 1,825 | 70.9% | +0.191% | +19.07% | YES |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.65, 4,657 trades, win rate 67.7%, avg return +0.048%/trade (+4.77% per 100 trades).  OOS win rate exceeds 66.7% breakeven -- **profitable on training-adjacent data**.

**Holdout (2025+, clean):** 6 trades, win rate 100.0%, avg return +1.500%/trade (+150.00% per 100 trades).  **Holdout breakeven exceeded.**

**Root cause of OOS/holdout gap:**  
OOS precision (~71%) reflects historical Sideways periods the model was
implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,
more volatile window (post-rate-hike cycle, AI bull market) where the
market moved directionally more often than the model expected.
With only ~304 holdout rows per ticker (~1.2 years), confidence intervals
are wide. Re-evaluate once 2025-2026 data accumulates.

**Best current candidate for cautious paper trading:**  
No holdout trades at this threshold.

---

*Generated by src/pipeline/10_backtest.py*
