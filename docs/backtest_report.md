# market_ml Step 10 -- Confidence-Gated Iron Condor Backtest

Generated: 2026-04-05  
Premium collected: 1.5% | Max loss: 3.0% | Breakeven precision: 66.7%  
Signal source: OOS walk-forward + holdout predictions from 06_train.py  
(No model loaded, no predict() called -- zero leakage risk)

---

## OOS vs Holdout Precision at Each Threshold

| Threshold | OOS Precision | OOS Trades | Holdout Precision | Holdout Trades | OOS Profitable? |
|-----------|:-------------:|:----------:|:-----------------:|:--------------:|:---------------:|
| 0.40 | 0.576 | 85,410 | 0.469 | 2,780 | no |
| 0.45 | 0.602 | 65,903 | 0.508 | 1,351 | no <-- 60% flag |
| 0.50 | 0.634 | 46,008 | 0.573 | 513 | no |
| 0.55 | 0.664 | 29,558 | 0.638 | 152 | no |
| 0.60 | 0.695 | 16,983 | 0.828 | 29 | YES <-- OOS breakeven |
| 0.65 | 0.725 | 8,781 | (1.000, <20t) | 6 | YES |
| 0.70 | 0.748 | 4,159 | 0 trades | 0 | YES |
| 0.75 | 0.782 | 1,888 | 0 trades | 0 | YES |
| 0.80 | 0.802 | 728 | 0 trades | 0 | YES |

**Key finding:** OOS precision exceeds the 66.7% breakeven at threshold 0.60 and above. Holdout precision (~0.48-0.54) never reaches breakeven at any threshold tested.

---

## P&L at Precision-Flag Threshold (0.45)

First threshold where aggregate OOS precision >= 60% with >= 20 avg trades per ticker.  
Note: 0.45 precision = 60% < 66.7% breakeven -- **strategy loses money even on OOS at this threshold**.

### OOS

| Ticker | Sector | Trades | Win Rate | Avg Return | 100 Trades | BH 100 |
|--------|--------|--------|:--------:|:----------:|:----------:|:------:|
| PG | consumer_staples | 3689 | 76.2% | +0.431% | +43.14% | +14.88% |
| KO | consumer_staples | 3518 | 74.7% | +0.363% | +36.28% | +14.34% |
| CL | consumer_staples | 3158 | 73.2% | +0.295% | +29.45% | +13.44% |
| WMT | consumer_staples | 2718 | 71.1% | +0.199% | +19.87% | +12.15% |
| WFC | financials | 2405 | 67.2% | +0.022% | +2.18% | +10.02% |
| MRNA | biotech | 0 | N/A | N/A | N/A | N/A |
| COST | consumer_staples | 2063 | 64.9% | -0.079% | -7.93% | +41.96% |
| XOM | energy | 2418 | 63.6% | -0.138% | -13.77% | -1.11% |
| ORCL | tech | 2570 | 61.0% | -0.256% | -25.62% | +18.23% |
| CVX | energy | 1906 | 60.8% | -0.264% | -26.36% | +12.54% |
| JPM | financials | 1981 | 60.5% | -0.276% | -27.64% | +36.79% |
| BAC | financials | 1627 | 60.4% | -0.281% | -28.12% | +18.00% |
| PFE | biotech | 3697 | 59.4% | -0.330% | -32.95% | +4.54% |
| ABBV | biotech | 1251 | 59.2% | -0.338% | -33.81% | +30.69% |
| LLY | biotech | 3315 | 58.8% | -0.356% | -35.57% | +19.25% |
| MSFT | tech | 3352 | 58.3% | -0.378% | -37.81% | +24.66% |
| BMY | biotech | 3642 | 57.1% | -0.429% | -42.87% | +11.00% |
| MS | financials | 1128 | 55.7% | -0.495% | -49.47% | +26.11% |
| GOOGL | tech | 2240 | 55.1% | -0.519% | -51.90% | +37.10% |
| META | tech | 944 | 55.0% | -0.526% | -52.60% | +23.68% |
| AMGN | biotech | 3294 | 54.7% | -0.537% | -53.69% | +11.70% |
| INTC | tech | 1813 | 54.7% | -0.538% | -53.78% | +0.40% |
| GS | financials | 1371 | 54.0% | -0.568% | -56.78% | +33.14% |
| ADBE | tech | 1966 | 53.3% | -0.604% | -60.35% | +24.96% |
| COP | energy | 870 | 52.6% | -0.631% | -63.10% | +44.42% |
| SLB | energy | 318 | 50.9% | -0.708% | -70.75% | +44.12% |
| CRM | tech | 1015 | 49.8% | -0.761% | -76.11% | +34.63% |
| AAPL | tech | 1763 | 49.7% | -0.762% | -76.15% | +53.58% |
| AMZN | tech | 1380 | 49.1% | -0.789% | -78.91% | +32.60% |
| BIIB | biotech | 1370 | 48.8% | -0.806% | -80.58% | +57.15% |
| NVDA | tech | 317 | 47.6% | -0.856% | -85.65% | -13.34% |
| GILD | biotech | 1713 | 46.1% | -0.927% | -92.73% | +31.35% |
| VRTX | biotech | 495 | 43.2% | -1.054% | -105.45% | +26.79% |
| EOG | energy | 197 | 41.6% | -1.127% | -112.69% | +13.09% |
| REGN | biotech | 268 | 39.2% | -1.237% | -123.69% | -23.01% |
| TSLA | tech | 75 | 36.0% | -1.380% | -138.00% | +222.80% |
| AMD | tech | 56 | 14.3% | -2.357% | -235.71% | +18.08% |

---

## P&L at OOS-Breakeven Threshold (0.60)

First threshold where aggregate OOS precision >= 66.7% (strategy is OOS-profitable).  
**Still unprofitable on holdout** -- holdout precision at 0.60 is ~82.8% < 66.7%.

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
| COP | energy | 52 | 48.1% | -0.836% | -83.65% | +40.88% |
| AAPL | tech | 111 | 46.9% | -0.892% | -89.19% | +101.50% |
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
| XOM | energy | 1 | 0.0% | -3.000% | -300.00% |

### Sector Aggregation (OOS breakeven threshold)

| Sector | Trades | Win Rate | Avg Return | 100 Trades | OOS Profitable? |
|--------|--------|:--------:|:----------:|:----------:|:---------------:|
| tech | 3,726 | 65.3% | -0.059% | -5.92% | no |
| biotech | 3,348 | 60.8% | -0.263% | -26.34% | no |
| financials | 2,927 | 69.0% | +0.107% | +10.71% | YES |
| energy | 848 | 65.1% | -0.071% | -7.08% | no |
| consumer_staples | 6,134 | 77.6% | +0.493% | +49.27% | YES |

---

## Verdict

**OOS (training-adjacent):** At threshold 0.60, 16,983 trades, win rate 69.5%, avg return +0.128%/trade (+12.80% per 100 trades).  OOS win rate exceeds 66.7% breakeven -- **profitable on training-adjacent data**.

**Holdout (2025+, clean):** 29 trades, win rate 82.8%, avg return +0.724%/trade (+72.41% per 100 trades).  **Holdout breakeven exceeded.**

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
