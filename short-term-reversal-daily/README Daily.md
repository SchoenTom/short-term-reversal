# Short-term Reversal Strategy Analysis (Daily Data)

Empirical analysis of the short-term reversal anomaly (Jegadeesh 1990) using US equity data from 1926-2024 at **daily frequency**.

## Quick Start

```bash
python main.py
```

Results are automatically saved to `output/`. ZIP files in the `data/` folder are extracted automatically.

---

## What This Project Does

This analysis investigates whether buying past losers and selling past winners still works as a trading strategy. The short-term reversal effect was first documented by Jegadeesh (1990), who found that stocks with poor performance over the past month tend to outperform in the following month.

We examine:
- Performance across different time periods
- Factor model decomposition (CAPM, Fama-French, Q-factors)
- Exposure to geopolitical risk
- Portfolio optimization through combination with residual momentum

The project follows the tasks outlined in the seminar project information sheet, using **daily returns** for higher-frequency analysis.

---

## Main Findings

### 1. The Anomaly Has Weakened But Remains Significant

Looking at CAPM alphas across different periods:

| Period | N | Alpha | t-stat | p-value | Significant? |
|--------|---|-------|--------|---------|--------------|
| Full Sample (1926-2024) | 25,901 | 6.03% | 13.92 | 0.0000 | Yes *** |
| Pre-1990 | 17,084 | 8.29% | 14.61 | 0.0000 | Yes *** |
| Post-1990 | 8,817 | 1.64% | 2.59 | 0.0096 | Yes *** |

Unlike the monthly analysis, daily data reveals that the alpha remains statistically significant even post-1990, though substantially reduced. The higher number of observations provides more statistical power to detect smaller effects.

### 2. Factor Models Tell Different Stories

Full sample results (1926-2024):

| Model | Alpha | t-stat | Adj. R² |
|-------|-------|--------|---------|
| CAPM | 6.03% | 13.92*** | 0.018 |
| FF3 | 6.10% | 14.10*** | 0.024 |
| FF5 | 4.31% | 10.21*** | 0.044 |
| Q4 | 4.28% | 9.71*** | 0.025 |
| Q5 | 4.15% | 9.37*** | 0.025 |

All models show highly significant alphas, even FF5 and Q5. The low adjusted R² values indicate that standard factors explain little of the strategy's variance, suggesting it captures something distinct from traditional risk exposures.

**Pre-1990 vs Post-1990:**

| Period | CAPM α | FF3 α | FF5 α | Q5 α |
|--------|--------|-------|-------|------|
| Pre-1990 | 8.29%*** | 8.42%*** | 8.40%*** | 8.88%*** |
| Post-1990 | 1.64%*** | 1.58%** | 1.33%** | 1.37%** |

The subperiod analysis confirms alphas remain significant in both periods across all models, though they have declined by approximately 80% after publication.

### 3. No Geopolitical Risk Exposure

Regressing daily returns on changes in the Geopolitical Risk Index (Caldara & Iacoviello 2022):

- **Coefficient:** 0.000000
- **t-statistic:** 0.12
- **p-value:** 0.90

The near-zero coefficient indicates the strategy has no meaningful exposure to geopolitical events. It's neither a hedge against geopolitical risk nor does it amplify such risks. Returns are essentially **orthogonal** to GPR changes.

### 4. Combining With Residual Momentum Works Exceptionally Well

Short-term reversal and residual momentum have a negative correlation (-0.122), which creates substantial diversification benefits when combined.

**Optimal portfolio:** 59% STR / 41% Residual Momentum

| Strategy | Return | Volatility | Sharpe |
|----------|--------|------------|--------|
| STR only | 4.56% | 3.26% | 1.398 |
| Residual Momentum only | 8.00% | 5.15% | 1.555 |
| Combined (59%/41%) | 5.97% | 2.68% | **2.230** |

The combined strategy improves the Sharpe ratio by **59.6%** relative to STR alone, while adding 1.41% annual return and reducing volatility by 0.59%. The combined portfolio shows strong alphas across all factor models (CAPM alpha of 5.98%, t=18.90).

---

## Performance Over Time

The strategy's performance has changed dramatically over the decades:

**Pre-1990 (1926-1989):**
- Annual return: 7.93%
- Volatility: 4.75%
- Sharpe ratio: 1.671
- Strong alphas across all models (>8% p.a.)

**Post-1990 (1990-2024):**
- Annual return: 1.56%
- Volatility: 3.74%
- Sharpe ratio: 0.418
- Reduced but still significant alphas (~1.3-1.6% p.a.)

This pattern is consistent with the publication effect documented by McLean & Pontiff (2016): anomalies tend to weaken after being published in academic journals as investors arbitrage them away. However, the daily data analysis reveals that significant (though diminished) alphas persist even in the post-publication era.

---

## Factor Exposures

The strategy shows interesting factor loadings:

**Full Sample (CAPM):**
- Market beta: -0.035 (t=-21.91***)

**Full Sample (FF3):**
- Market: -0.038 (t=-23.10***)
- Size (SMB): -0.035 (t=-12.06***)
- Value (HML): -0.002 (not significant)

**Full Sample (FF5):**
- Market: -0.022 (t=-12.04***)
- Size (SMB): -0.045 (t=-13.54***)
- Value (HML): +0.034 (t=9.88***)
- Profitability (RMW): +0.028 (t=6.04***)
- Investment (CMA): -0.002 (not significant)

The negative market beta makes sense for a reversal strategy - it bets against recent market movements. The negative size exposure indicates a large-cap tilt. In the FF5 model, positive value and profitability loadings emerge, suggesting the strategy aligns with quality characteristics when controlling for other factors.

---

## Visualizations

The analysis generates five figures showing different aspects of the strategy:

1. **Cumulative wealth (1926-2024):** Shows the long-run performance of $1 invested in the strategy
2. **Pre vs Post publication:** Compares performance before and after Jegadeesh (1990)
3. **STR vs Combined:** Demonstrates the benefits of combining with residual momentum
4. **All three strategies:** Shows STR, residual momentum, and combined
5. **Rolling returns:** 252-day (annual) rolling returns over time

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
- `figure5_rolling_returns.png` - Rolling 252-day returns

---

## Interpretation

The daily data analysis provides stronger statistical evidence than monthly data due to the larger sample size (25,901 vs ~1,200 observations). Key insights:

1. **Persistent but diminished alpha:** Unlike monthly results where post-1990 alphas often become insignificant, daily data reveals that small but statistically significant alphas persist even after publication.

2. **Robust factor model alphas:** All five factor models (CAPM through Q5) show significant alphas, suggesting the reversal effect captures something beyond standard risk factors.

3. **Excellent diversification potential:** The negative correlation with residual momentum (-0.122) enables a combined strategy with a Sharpe ratio of 2.23 - exceptional by any standard.

4. **No geopolitical sensitivity:** The strategy is effectively neutral to geopolitical risk, making it useful in uncertain times.

For practical purposes, while the standalone short-term reversal strategy has weakened considerably since 1990 (Sharpe of 0.42 vs 1.67 pre-1990), it remains valuable as a diversification tool. The combined STR/Residual Momentum portfolio achieves risk-adjusted returns that exceed either strategy alone.

---

## Data Sources

- **Strategy returns:** Global Factor Data (daily, equal-weighted, US stocks)
- **Fama-French factors:** Kenneth French Data Library (daily)
- **Q-factors:** Hou, Xue, Zhang global-q.org (daily)
- **GPR Index:** Caldara & Iacoviello (2022) (daily)

---

## Technical Notes

- **Annualization:** Daily returns are annualized using 252 trading days per year
- **Rolling windows:** 252-day windows for rolling return calculations
- **Minimum observations:** 504 days (~2 years) required for subperiod regressions
- **ZIP extraction:** The script automatically extracts any .zip files in the data directory

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
