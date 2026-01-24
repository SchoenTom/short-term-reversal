# Short-term Reversal Strategy Analysis

**Semester Project: Stock Markets in the Age of Big Data**  
University of Konstanz | Winter 2025/2026

An empirical investigation of the short-term reversal anomaly (Jegadeesh 1990) examining whether buying past losers and selling past winners remains profitable in modern markets.

---

## Key Finding

The short-term reversal effect has weakened substantially since its academic publication but hasn't disappeared entirely. Combining it with residual momentum creates a portfolio with a Sharpe ratio exceeding 2.0.

---

## Project Structure

```
short-term-reversal/
├── short-term-reversal-monthly/     # Monthly frequency analysis
│   ├── main.py                      # Analysis script
│   ├── data/                        # Input data files
│   └── output/                      # Results (tables & figures)
│
└── short-term-reversal-daily/       # Daily frequency analysis
    ├── main.py                      # Analysis script
    ├── data/                        # Input data files
    └── output/                      # Results (tables & figures)
```

Both implementations follow identical methodology, differing only in data frequency. The daily analysis provides greater statistical power (25,901 vs 1,182 observations), while the monthly analysis aligns with the original Jegadeesh (1990) study.

---

## Quick Start

```bash
# Monthly analysis
cd short-term-reversal-monthly
python main.py

# Daily analysis
cd short-term-reversal-daily
python main.py
```

Results are saved to the respective `output/` directories.

---

## Analysis Overview

The project addresses four core questions from the seminar information sheet:

| Task | Question | Finding |
|------|----------|---------|
| **Task 2** | Performance & Factor Models | Alpha declines ~80% post-publication; FF5/Q5 models absorb most returns |
| **Task 4** | Geopolitical Risk Exposure | Zero correlation with GPR—strategy is orthogonal to geopolitical events |
| **Task 5** | Combined Strategy | 49% STR / 51% ResMom combination achieves Sharpe > 2.0 |

---

## Main Results

### Publication Effect (Pre vs Post 1990)

The strategy's performance declined sharply after Jegadeesh (1990) was published:

| Period | Annual Return | Sharpe Ratio | CAPM Alpha |
|--------|--------------|--------------|------------|
| **Pre-1990** | 7.93% | 1.67 | 8.29%*** |
| **Post-1990** | 1.56% | 0.42 | 1.64%** |
| **Decline** | -80% | -75% | -80% |

This pattern aligns with McLean & Pontiff (2016): anomalies weaken as arbitrage capital flows in after publication.

### Factor Model Decomposition

Full sample alpha persistence across increasingly comprehensive models:

| Model | Alpha (ann.) | t-statistic | Interpretation |
|-------|-------------|-------------|----------------|
| CAPM | 6.03% | 13.92*** | Strong raw alpha |
| FF3 | 6.10% | 14.10*** | Not explained by size/value |
| FF5 | 4.31% | 10.21*** | Partially explained by profitability/investment |
| Q5 | 4.15% | 9.37*** | Investment-based model leaves alpha |

Even the most comprehensive factor models cannot fully explain STR returns.

### Portfolio Optimization

Combining STR with residual momentum exploits their negative correlation (-0.12):

| Strategy | Return | Volatility | Sharpe |
|----------|--------|------------|--------|
| STR only | 4.56% | 3.26% | 1.40 |
| ResMom only | 8.00% | 5.15% | 1.56 |
| **Combined (59/41)** | 5.97% | 2.68% | **2.23** |

The combined strategy achieves a 60% improvement in risk-adjusted returns.

---

## Output Files

### Tables
| File | Description |
|------|-------------|
| `table1_performance.csv` | Summary statistics by period |
| `table2_factor_models.csv` | Factor model regressions |
| `table2b_factor_models_subperiods.csv` | Pre/post 1990 comparison |
| `table2c_factor_exposures.csv` | Factor betas |
| `table4_gpr.csv` | Geopolitical risk regressions |
| `table5_portfolio.csv` | Combined strategy performance |

### Figures
| File | Description |
|------|-------------|
| `figure1_str_wealth.png` | Cumulative returns 1926-2024 |
| `figure2_str_publication_effect.png` | Pre vs post publication |
| `figure3_str_vs_combined.png` | STR vs optimized portfolio |
| `figure5_rolling_returns.png` | Rolling annual returns |
| `figure6_sortino_ratio.png` | Downside risk comparison |
| `figure7_calmar_ratio.png` | Drawdown-adjusted returns |

---

## Methodology

### Data Sources
- **Strategy Returns:** Global Factor Data (globalfactordata.com)
- **Fama-French Factors:** Kenneth French Data Library
- **Q-Factors:** Hou, Xue, Zhang (global-q.org)
- **GPR Index:** Caldara & Iacoviello (2022)

### Statistical Methods
- OLS regression with standard errors
- Annualization: 12× for monthly, 252× for daily
- Portfolio optimization via Sharpe ratio maximization

---

## Interpretation

The evidence suggests a **behavioral origin** for short-term reversal rather than risk compensation:

1. If returns compensated systematic risk, they would persist
2. Instead, alphas decay after publication → arbitrage story
3. Low factor model R² → captures something beyond standard risks
4. No GPR exposure → not compensation for geopolitical risk

The pattern is consistent with the **Adaptive Markets Hypothesis** (Lo 2004): market efficiency varies over time as arbitrageurs discover and exploit anomalies.

---

## Dependencies

```
pandas
numpy
matplotlib
scipy
xlrd  # for GPR Excel files
```

---

## References

- Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881-898.
- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- Hou, K., Xue, C., & Zhang, L. (2015). Digesting anomalies: An investment approach. *Review of Financial Studies*, 28(3), 650-705.
- McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5-32.
- Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194-1225.
- Lo, A. W. (2004). The adaptive markets hypothesis. *Journal of Portfolio Management*, 30(5), 15-29.

---

## License

Academic use only. University of Konstanz, Winter 2025/2026.
