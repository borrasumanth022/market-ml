# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-04-05  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.554 | 98,608 | 0.445 | 3,252 | no |
| 0.45 | 0.579 | 75,471 | 0.478 | 1,579 | no |
| 0.50 | 0.611 | 51,719 | 0.531 | 604 | no <-- 60% flag |
| 0.55 | 0.642 | 32,519 | 0.561 | 198 | no |
| 0.60 | 0.675 | 18,385 | 0.692 | 39 | YES <-- OOS breakeven |
| 0.65 | 0.708 | 9,323 | (0.750, <20t) | 8 | YES |
| 0.70 | 0.732 | 4,335 | 0 trades | 0 | YES |
| 0.75 | 0.773 | 1,932 | 0 trades | 0 | YES |
| 0.80 | 0.801 | 732 | 0 trades | 0 | YES |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.60 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.50)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.50 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| PG | consumer_staples | 3109 | 77.9% | +0.504% | +50.42% | +14.14% |
| KO | consumer_staples | 2940 | 75.8% | +0.409% | +40.87% | +14.38% |
| CL | consumer_staples | 2523 | 74.1% | +0.333% | +33.35% | +13.99% |
| WMT | consumer_staples | 2061 | 73.3% | +0.299% | +29.91% | +10.66% |
| WFC | financials | 1996 | 69.1% | +0.111% | +11.12% | +13.09% |
| XOM | energy | 1615 | 66.9% | +0.012% | +1.21% | +4.69% |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| COST | consumer_staples | 1356 | 66.6% | -0.003% | -0.33% | +40.68% |
| ORCL | tech | 1811 | 64.6% | -0.093% | -9.28% | +12.56% |
| JPM | financials | 1491 | 63.3% | -0.151% | -15.09% | +37.38% |
| BAC | financials | 1159 | 62.7% | -0.177% | -17.73% | +18.40% |
| ABBV | biotech | 670 | 61.3% | -0.240% | -23.96% | +41.85% |
| CVX | energy | 1227 | 61.2% | -0.246% | -24.57% | +15.92% |
| PFE | biotech | 2762 | 61.1% | -0.250% | -24.98% | +4.82% |
| LLY | biotech | 2440 | 60.6% | -0.274% | -27.42% | +26.67% |
| MSFT | tech | 2707 | 60.2% | -0.292% | -29.20% | +25.77% |
| INTC | tech | 1159 | 59.8% | -0.309% | -30.93% | -2.04% |
| MS | financials | 678 | 59.4% | -0.325% | -32.52% | +23.76% |
| BMY | biotech | 2737 | 59.0% | -0.345% | -34.47% | +23.26% |
| GOOGL | tech | 1589 | 58.5% | -0.366% | -36.63% | +33.66% |
| META | tech | 649 | 57.2% | -0.428% | -42.76% | +37.67% |
| ADBE | tech | 1371 | 56.9% | -0.440% | -43.98% | +36.78% |
| AMGN | biotech | 2214 | 56.1% | -0.476% | -47.56% | +10.65% |
| GS | financials | 895 | 55.5% | -0.501% | -50.11% | +42.34% |
| NVDA | tech | 92 | 54.4% | -0.554% | -55.43% | -39.82% |
| CRM | tech | 553 | 53.9% | -0.575% | -57.50% | +23.17% |
| COP | energy | 447 | 53.7% | -0.584% | -58.39% | +61.68% |
| SLB | energy | 124 | 53.2% | -0.605% | -60.48% | +55.13% |
| AAPL | tech | 987 | 51.5% | -0.684% | -68.39% | +68.81% |
| AMZN | tech | 758 | 51.4% | -0.685% | -68.47% | +30.57% |
| BIIB | biotech | 735 | 50.1% | -0.747% | -74.69% | +83.38% |
| TSLA | tech | 8 | 50.0% | -0.750% | -75.00% | +155.89% |
| GILD | biotech | 834 | 48.2% | -0.831% | -83.09% | +30.15% |
| KLAC | semiconductors | 1148 | 45.6% | -0.950% | -94.99% | +20.68% |
| TSM | semiconductors | 1069 | 43.7% | -1.034% | -103.41% | -0.77% |
| ASML | semiconductors | 1405 | 43.2% | -1.056% | -105.59% | +35.05% |
| AMAT | semiconductors | 1242 | 40.3% | -1.185% | -118.48% | +36.56% |
| LRCX | semiconductors | 847 | 40.0% | -1.199% | -119.89% | +28.17% |
| REGN | biotech | 74 | 36.5% | -1.358% | -135.81% | -33.66% |
| EOG | energy | 58 | 36.2% | -1.371% | -137.07% | -65.81% |
| VRTX | biotech | 166 | 32.5% | -1.536% | -153.61% | +46.60% |
| AMD | tech | 13 | 7.7% | -2.654% | -265.38% | -162.56% |

---

## P&L at OOS-Breakeven Threshold (0.60)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.60 is ~69.2% < 66.7%.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| NVDA | tech | 1 | 100.0% | +1.500% | +150.00% | -24.70% |
| REGN | biotech | 1 | 100.0% | +1.500% | +150.00% | -45.61% |
| PG | consumer_staples | 1934 | 81.5% | +0.667% | +66.70% | +20.74% |
| KO | consumer_staples | 1538 | 76.9% | +0.461% | +46.13% | +20.57% |
| CL | consumer_staples | 1338 | 76.5% | +0.444% | +44.39% | +15.23% |
| WMT | consumer_staples | 873 | 75.3% | +0.387% | +38.66% | +10.64% |
| WFC | financials | 1308 | 71.9% | +0.234% | +23.39% | +9.55% |
| COST | consumer_staples | 451 | 71.2% | +0.203% | +20.29% | +35.90% |
| ORCL | tech | 563 | 69.8% | +0.141% | +14.12% | -12.16% |
| JPM | financials | 703 | 69.7% | +0.137% | +13.66% | +39.43% |
| META | tech | 278 | 68.7% | +0.092% | +9.17% | +50.41% |
| BAC | financials | 492 | 68.7% | +0.091% | +9.15% | +6.61% |
| XOM | energy | 454 | 67.6% | +0.043% | +4.30% | +14.33% |
| GOOGL | tech | 368 | 67.1% | +0.020% | +2.04% | +34.59% |
| AMD | tech | 0 | N/A | N/A | N/A | N/A |
| TSLA | tech | 0 | N/A | N/A | N/A | N/A |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| PFE | biotech | 856 | 65.9% | -0.035% | -3.50% | +4.77% |
| CVX | energy | 323 | 65.3% | -0.060% | -6.04% | +42.54% |
| MSFT | tech | 1391 | 65.3% | -0.062% | -6.25% | +15.80% |
| INTC | tech | 201 | 64.7% | -0.090% | -8.96% | -1.28% |
| CRM | tech | 169 | 64.5% | -0.098% | -9.76% | -22.35% |
| MS | financials | 150 | 64.0% | -0.120% | -12.00% | +23.23% |
| ADBE | tech | 477 | 63.1% | -0.160% | -16.04% | +21.60% |
| LLY | biotech | 736 | 62.4% | -0.194% | -19.36% | +20.36% |
| AMZN | tech | 167 | 61.7% | -0.225% | -22.46% | +27.75% |
| BMY | biotech | 1012 | 60.9% | -0.261% | -26.09% | +20.91% |
| GS | financials | 274 | 57.3% | -0.421% | -42.15% | +48.57% |
| SLB | energy | 16 | 56.2% | -0.469% | -46.88% | +158.42% |
| VRTX | biotech | 9 | 55.6% | -0.500% | -50.00% | +72.42% |
| AMGN | biotech | 454 | 54.4% | -0.552% | -55.18% | +1.28% |
| ABBV | biotech | 68 | 52.9% | -0.618% | -61.76% | +88.67% |
| BIIB | biotech | 92 | 51.1% | -0.701% | -70.11% | +84.33% |
| GILD | biotech | 120 | 50.8% | -0.713% | -71.25% | -24.25% |
| KLAC | semiconductors | 267 | 48.7% | -0.809% | -80.90% | -1.41% |
| COP | energy | 52 | 48.1% | -0.836% | -83.65% | +40.88% |
| AAPL | tech | 111 | 46.9% | -0.892% | -89.19% | +101.50% |
| ASML | semiconductors | 426 | 45.1% | -0.972% | -97.18% | +35.86% |
| AMAT | semiconductors | 325 | 41.2% | -1.145% | -114.46% | -25.26% |
| TSM | semiconductors | 206 | 39.8% | -1.209% | -120.87% | +29.45% |
| LRCX | semiconductors | 178 | 36.0% | -1.382% | -138.20% | +2.36% |
| EOG | energy | 3 | 0.0% | -3.000% | -300.00% | -56.01% |

### Holdout (2024+)

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades |
|--------|--------|--------|:--------:|:----------:|:----------:|
| JPM | financials | 2 | 100.0% | +1.500% | +150.00% |
| BAC | financials | 6 | 100.0% | +1.500% | +150.00% |
| MS | financials | 2 | 100.0% | +1.500% | +150.00% |
| CVX | energy | 2 | 100.0% | +1.500% | +150.00% |
| KO | consumer_staples | 1 | 100.0% | +1.500% | +150.00% |
| WFC | financials | 6 | 83.3% | +0.750% | +75.00% |
| AAPL | tech | 0 | N/A | N/A | N/A |
| MSFT | tech | 9 | 66.7% | +0.000% | +0.00% |
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
| COP | energy | 0 | N/A | N/A | N/A |
| SLB | energy | 0 | N/A | N/A | N/A |
| EOG | energy | 0 | N/A | N/A | N/A |
| PG | consumer_staples | 0 | N/A | N/A | N/A |
| WMT | consumer_staples | 0 | N/A | N/A | N/A |
| COST | consumer_staples | 0 | N/A | N/A | N/A |
| CL | consumer_staples | 0 | N/A | N/A | N/A |
| TSM | semiconductors | 0 | N/A | N/A | N/A |
| LRCX | semiconductors | 0 | N/A | N/A | N/A |
| ASML | semiconductors | 4 | 50.0% | -0.750% | -75.00% |
| KLAC | semiconductors | 2 | 50.0% | -0.750% | -75.00% |
| XOM | energy | 1 | 0.0% | -3.000% | -300.00% |
| AMAT | semiconductors | 4 | 0.0% | -3.000% | -300.00% |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 3,726 | 65.3% | -0.059% | -5.92% | no |
| biotech | 3,348 | 60.8% | -0.263% | -26.34% | no |
| financials | 2,927 | 69.0% | +0.107% | +10.71% | YES |
| energy | 848 | 65.1% | -0.071% | -7.08% | no |
| consumer_staples | 6,134 | 77.6% | +0.493% | +49.27% | YES |
| semiconductors | 1,402 | 42.9% | -1.068% | -106.78% | no |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.60, 18,385 trades, win rate 67.5%, avg return +0.037%/trade (+3.68% per 100 trades).  OOS win rate exceeds 66.7% breakeven -- **profitable on training-adjacent data**.

**Holdout (2025+, clean):** 39 trades, win rate 69.2%, avg return +0.115%/trade (+11.54% per 100 trades).  **Holdout breakeven exceeded.**

**Root cause of OOS/holdout gap:**  
OOS precision (~71%) reflects historical Sideways periods the model was
implicitly calibrated on (2000-2023). The holdout (2024+) is a shorter,
more volatile window (post-rate-hike cycle, AI bull market) where the
market moved directionally more often than the model expected.
With only ~304 holdout rows per ticker (~1.2 years), confidence intervals
are wide. Re-evaluate once 2025-2026 data accumulates.

**Best current candidate for cautious paper trading:**  
- **BAC** (financials): holdout win rate 100.0% (6 trades), avg +1.500%/trade
- **WFC** (financials): holdout win rate 83.3% (6 trades), avg +0.750%/trade
- **MSFT** (tech): holdout win rate 66.7% (9 trades), avg +0.000%/trade

---

*Generated by src/pipeline/10_backtest.py*
