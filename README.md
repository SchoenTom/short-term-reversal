# Short-term Reversal Strategy Analysis

Empirical analysis of the short-term reversal anomaly (Jegadeesh 1990) using US equity data from 1926-2024.

## Quick Start

```bash
python main.py
```

Results are automatically saved to `output/`.

---

## What This Project Does

This analysis investigates whether buying past losers and selling past winners still works as a trading strategy. The short-term reversal effect was first documented by Jegadeesh (1990), who found that stocks with poor performance over the past month tend to outperform in the following month.

We examine:
- Performance across different time periods
- Factor model decomposition (CAPM, Fama-French, Q-factors)
- Exposure to geopolitical risk
- Portfolio optimization through combination with residual momentum

The project follows the tasks outlined in the seminar project information sheet.

---

## Main Findings

### 1. The Anomaly Has Weakened But Not Disappeared

Looking at CAPM alphas across different periods:

| Period | N | Alpha | t-stat | p-value | Significant? |
|--------|---|-------|--------|---------|--------------|
| Full Sample (1926-2024) | 1182 | 1.66% | 3.11 | 0.0019 | Yes *** |
| Pre-1990 | 762 | 2.19% | 3.17 | 0.0016 | Yes *** |
| Post-1990 | 420 | 0.53% | 0.65 | 0.5183 | No |

The alpha in recent decades remains positive but is no longer statistically significant. This suggests the strategy still generates small excess returns, but they're not reliable enough to bet on.

### 2. Factor Models Tell Different Stories

Full sample results (1926-2024):

| Model | Alpha | t-stat | Adj. R² |
|-------|-------|--------|---------|
| CAPM | 1.66% | 3.11*** | 0.019 |
| FF3 | 1.80% | 3.38*** | 0.030 |
| FF5 | 0.27% | 0.46 | 0.011 |
| Q5 | 0.71% | 1.05 | 0.003 |

The five-factor models (FF5 and Q5) explain away most of the alpha, which makes sense since they account for profitability and investment patterns. The low adjusted R² values show that standard factors don't explain much of the strategy's variance, suggesting it captures something different from traditional risk exposures.

**Pre-1990 vs Post-1990:**

The subperiod analysis reveals that all models show significant alphas before 1990, but none are significant afterwards. Even the FF5 and Q5 models can't explain the early-period returns, but the alphas themselves have weakened substantially in recent decades.

### 3. No Geopolitical Risk Exposure

Regressing monthly returns on changes in the Geopolitical Risk Index (Caldara & Iacoviello 2022):

- **Coefficient:** -0.000009
- **t-statistic:** -0.51
- **p-value:** 0.61

The near-zero coefficient indicates the strategy has no meaningful exposure to geopolitical events. It's neither a hedge against geopolitical risk nor does it amplify such risks.

### 4. Combining With Residual Momentum Works Well

Short-term reversal and residual momentum have a slight negative correlation (-0.08), which creates diversification benefits when combined.

**Optimal portfolio:** 49% STR / 51% Residual Momentum

| Strategy | Return | Volatility | Sharpe |
|----------|--------|------------|--------|
| STR only | 0.83% | 4.22% | 0.197 |
| Residual Momentum only | 4.86% | 9.19% | 0.529 |
| Combined | 2.89% | 4.96% | 0.582 |

The combined strategy improves the Sharpe ratio by 195% relative to STR alone, while adding 2.05% annual return. Interestingly, the combined portfolio shows strong alphas even in modern factor models (CAPM alpha of 3.09%, t=5.23), though the Q5 model fully explains these returns.

---

## Performance Over Time

The strategy's performance has changed dramatically over the decades:

**Pre-1990 (1926-1989):**
- Annual return: 1.76%
- Sharpe ratio: 0.315
- Strong alphas across all models

**Post-1990 (1990-2024):**
- Annual return: 0.54%
- Sharpe ratio: 0.114
- No significant alphas

This pattern is consistent with the publication effect documented by McLean & Pontiff (2016): anomalies tend to weaken after being published in academic journals as investors arbitrage them away. However, the fact that alphas remain positive (though insignificant) suggests the effect hasn't completely disappeared.

---

## Factor Exposures

The strategy shows interesting factor loadings:

**Full Sample (CAPM):**
- Market beta: -0.041 (t=-4.89***)

**Full Sample (FF3):**
- Market: -0.034 (t=-3.79***)
- Size (SMB): 0.004 (not significant)
- Value (HML): -0.050 (t=-3.97***)

The negative market beta makes sense for a reversal strategy - it bets against recent market movements. The negative value exposure suggests the strategy tilts toward growth stocks, which tend to show stronger short-term reversal patterns.

---

## Visualizations

The analysis generates five figures showing different aspects of the strategy:

1. **Cumulative wealth (1926-2024):** Shows the long-run performance of $1 invested in the strategy
2. **Pre vs Post publication:** Compares performance before and after Jegadeesh (1990)
3. **STR vs Combined:** Demonstrates the benefits of combining with residual momentum
4. **All three strategies:** Shows STR, residual momentum, and combined (note: residual momentum dominates the scale)
5. **Rolling returns:** 12-month rolling returns over time

---

## Output Files

All results are saved to the `output/` directory:

**Performance & Factor Models:**
- `table1_performance.csv` - Summary statistics by period
- `table2_factor_models.csv` - Factor model results (full sample)
- `table2b_factor_models_subperiods.csv` - Factor models for pre/post 1990
- `table2c_factor_exposures.csv` - Factor betas (full sample)
- `table2d_factor_exposures_subperiods.csv` - Factor betas by period

**Geopolitical Risk:**
- `table4_gpr.csv` - GPR regression results
- `table4b_gpr_combined.csv` - GPR exposure comparison (STR vs combined)

**Portfolio Optimization:**
- `table5_portfolio.csv` - Performance comparison
- `table5b_combined_factor_models.csv` - Factor models for combined strategy
- `table5c_combined_exposures.csv` - Factor exposures for combined strategy

**Figures:**
- `figure1_str_wealth.png` - STR cumulative returns
- `figure2_str_publication_effect.png` - Pre vs post 1990 comparison
- `figure3_str_vs_combined.png` - STR vs combined portfolio
- `figure4_all_strategies.png` - All three strategies
- `figure5_rolling_returns.png` - Rolling 12-month returns

---

## Interpretation

The evidence points to a behavioral explanation for the short-term reversal effect rather than a risk-based one:

1. **Discovery (pre-1970s):** The pattern existed but wasn't widely known
2. **Exploitation (1970s-1990s):** Academic research documented it, arbitrage increased
3. **Attenuation (post-1990):** After Jegadeesh (1990), the anomaly weakened substantially

If the returns were compensation for systematic risk, they should persist over time. Instead, we see them decay after publication, which is more consistent with behavioral biases being arbitraged away.

That said, the alphas haven't completely vanished - they're just no longer statistically significant. This might reflect limits to arbitrage or the fact that some behavioral patterns persist even when known.

For practical purposes, the short-term reversal strategy alone doesn't seem attractive in the modern era. The returns are too small and unreliable, especially after accounting for transaction costs. However, combining it with other strategies (like residual momentum) can improve risk-adjusted performance through diversification.

---

## Data Sources

- **Strategy returns:** Global Factor Data (monthly, value-weighted, US stocks)
- **Fama-French factors:** Kenneth French Data Library
- **Q-factors:** Hou, Xue, Zhang global-q.org
- **GPR Index:** Caldara & Iacoviello (2022)

---

## References

Blitz, D., Huij, J., & Martens, M. (2011). Residual momentum. *Journal of Empirical Finance*, 18(3), 506-521.

Caldara, D., & Iacoviello, M. (2022). Measuring geopolitical risk. *American Economic Review*, 112(4), 1194-1225.

Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

Hou, K., Xue, C., & Zhang, L. (2015). Digesting anomalies: An investment approach. *Review of Financial Studies*, 28(3), 650-705.

Hou, K., Xue, C., & Zhang, L. (2021). Replicating anomalies. *Review of Financial Studies*, 33(5), 2019-2133.

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881-898.

Lo, A. W. (2004). The adaptive markets hypothesis. *Journal of Portfolio Management*, 30(5), 15-29.

McLean, R. D., & Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5-32.
