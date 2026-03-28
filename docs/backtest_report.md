# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-03-28  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.612 | 20,540 | 0.477 | 1,464 | no <-- 60% flag |
| 0.45 | 0.658 | 15,356 | 0.502 | 933 | no |
| 0.50 | 0.717 | 10,442 | 0.525 | 514 | YES <-- OOS breakeven |
| 0.55 | 0.782 | 6,433 | 0.527 | 296 | YES |
| 0.60 | 0.846 | 3,634 | 0.554 | 177 | YES |
| 0.65 | 0.879 | 1,988 | 0.539 | 115 | YES |
| 0.70 | 0.931 | 909 | 0.567 | 67 | YES |
| 0.75 | 0.958 | 333 | 0.520 | 25 | YES |
| 0.80 | 0.951 | 61 | (0.538, <20t) | 13 | YES |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.50 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.40)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.40 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| MRNA | biotech | 4 | 100.0% | +1.500% | +150.00% | -63.60% |
| MSFT | tech | 3549 | 64.2% | -0.110% | -11.03% | +29.21% |
| VRTX | biotech | 1667 | 63.0% | -0.166% | -16.56% | +6.27% |
| META | tech | 1019 | 62.7% | -0.178% | -17.81% | +20.45% |
| GOOGL | tech | 2429 | 61.3% | -0.241% | -24.15% | +29.95% |
| LLY | biotech | 4608 | 61.1% | -0.250% | -25.00% | +24.02% |
| REGN | biotech | 1072 | 60.0% | -0.301% | -30.08% | +43.86% |
| AMZN | tech | 1491 | 60.0% | -0.302% | -30.18% | +35.24% |
| AAPL | tech | 1834 | 59.3% | -0.330% | -33.04% | +54.65% |
| NVDA | tech | 501 | 58.9% | -0.350% | -35.03% | +0.85% |
| BIIB | biotech | 2366 | 58.0% | -0.391% | -39.05% | +50.74% |

---

## P&L at OOS-Breakeven Threshold (0.50)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.50 is ~52.5% < 66.7%.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| REGN | biotech | 265 | 83.0% | +0.736% | +73.58% | +21.98% |
| META | tech | 498 | 76.1% | +0.425% | +42.47% | +14.19% |
| VRTX | biotech | 838 | 75.4% | +0.394% | +39.38% | -2.78% |
| AMZN | tech | 551 | 74.4% | +0.348% | +34.85% | +28.62% |
| BIIB | biotech | 820 | 73.0% | +0.287% | +28.72% | +40.36% |
| GOOGL | tech | 1196 | 71.6% | +0.221% | +22.07% | +14.42% |
| MSFT | tech | 2383 | 71.3% | +0.210% | +21.02% | +25.90% |
| AAPL | tech | 798 | 70.9% | +0.192% | +19.17% | +38.87% |
| NVDA | tech | 86 | 68.6% | +0.087% | +8.72% | +20.82% |
| LLY | biotech | 3007 | 68.6% | +0.086% | +8.58% | +20.84% |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |

### Holdout (2024+)

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |
|--------|--------|--------|:--------:|:----------:|:----------:|
| GOOGL | tech | 9 | 77.8% | +0.500% | +50.00% |
| BIIB | biotech | 4 | 75.0% | +0.375% | +37.50% |
| NVDA | tech | 0 | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A |
| VRTX | biotech | 215 | 57.2% | -0.426% | -42.56% |
| REGN | biotech | 43 | 55.8% | -0.488% | -48.84% |
| AMZN | tech | 12 | 50.0% | -0.750% | -75.00% |
| MSFT | tech | 134 | 49.2% | -0.784% | -78.36% |
| LLY | biotech | 44 | 47.7% | -0.852% | -85.23% |
| AAPL | tech | 48 | 39.6% | -1.219% | -121.87% |
| META | tech | 5 | 20.0% | -2.100% | -210.00% |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 5,512 | 72.0% | +0.241% | +24.11% | YES |
| biotech | 4,930 | 71.3% | +0.207% | +20.66% | YES |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.50, 10,442 trades, win rate 71.7%, avg return +0.225%/trade (+22.48% per 100 trades).  
OOS win rate exceeds the 66.7% breakeven -- **profitable on training-adjacent data**.

**Holdout (2024+, clean):** 514 trades, win rate 52.5%, avg return -0.636%/trade (-63.62% per 100 trades).  
Holdout win rate 52.5% < 66.7% breakeven -- **not yet profitable on genuinely unseen data**.

**Root cause of OOS/holdout gap:**  
OOS precision (~71%) reflects historical Sideways periods the model was
implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,
more volatile window (post-rate-hike cycle, AI bull market) where the
market moved directionally more often than the model expected.
With only ~304 holdout rows per ticker (~1.2 years), confidence intervals
are wide. Re-evaluate once 2025-2026 data accumulates.

**Best current candidate for cautious paper trading:**  
- **GOOGL** (tech): holdout win rate 77.8% (9 trades), avg +0.500%/trade

---

*Generated by src/pipeline/10_backtest.py*
