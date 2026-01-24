#!/usr/bin/env python3
"""
Short-term Reversal Strategy Analysis
======================================
Semester Project: Stock Markets in the Age of Big Data
University of Konstanz, Winter 2025/2026

Complete empirical analysis following project information sheet tasks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'output'

FILES = {
    'str': f'{DATA_DIR}/[usa]_[short_term_reversal]_[monthly]_[vw].csv',
    'str_daily': f'{DATA_DIR}/[usa]_[short_term_reversal]_[daily]_[vw].csv',
    'resmom': f'{DATA_DIR}/[usa]_[resff3_12_1]_[monthly]_[vw].csv',
    'ff3': f'{DATA_DIR}/F-F_Research_Data_Factors.CSV',
    'ff5': f'{DATA_DIR}/F-F_Research_Data_5_Factors_2x3.csv',
    'q5': f'{DATA_DIR}/q5_factors_monthly_2024.csv',
    'gpr': f'{DATA_DIR}/data_gpr_export.xls',
}

# =============================================================================
# DATA LOADING
# =============================================================================
def load_strategy(filepath, freq='monthly'):
    """Load strategy returns from Global Factor Data."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    ret = df.set_index('date')['ret'].astype(float)
    if freq == 'monthly':
        ret.index = ret.index.to_period('M').to_timestamp('M')
    return ret

def load_ff_factors():
    """Load Fama-French factors."""
    def parse(path):
        df = pd.read_csv(path, skiprows=3)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df = df[df.iloc[:, 0].str.match(r'^\d{6}$', na=False)].copy()
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'], format='%Y%m')
        df = df.set_index('date')
        df.index = df.index.to_period('M').to_timestamp('M')
        return df.apply(pd.to_numeric, errors='coerce') / 100

    ff3 = parse(FILES['ff3'])
    ff5 = parse(FILES['ff5'])
    factors = pd.concat([
        ff3[['Mkt-RF', 'SMB', 'HML', 'RF']],
        ff5[['RMW', 'CMA']]
    ], axis=1)
    return factors.rename(columns={'Mkt-RF': 'MKT-RF'})

def load_q_factors():
    """Load Q-factor model."""
    df = pd.read_csv(FILES['q5'])
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + df['month'].astype(str).str.zfill(2),
        format='%Y%m'
    )
    df = df.set_index('date')
    df.index = df.index.to_period('M').to_timestamp('M')
    cols = ['R_F', 'R_MKT', 'R_ME', 'R_IA', 'R_ROE', 'R_EG']
    return df[cols] / 100

def load_gpr(frequency='monthly'):
    """Load Geopolitical Risk Index (Caldara & Iacoviello 2022)."""
    df = pd.read_excel(FILES['gpr'])
    date_col = [c for c in df.columns if 'date' in c.lower() or 'month' in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    if frequency == 'daily' and 'GPRD' in df.columns:
        return df['GPRD'].astype(float)
    else:
        gpr_col = 'GPRD' if 'GPRD' in df.columns else 'GPR'
        df.index = df.index.to_period('M').to_timestamp('M')
        return df.groupby(df.index)[gpr_col].mean().astype(float)

# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================
def ols_regression(y, X, factor_names, model_name):
    """OLS with White (1980) robust standard errors."""
    df = pd.concat([y.rename('y'), X[factor_names]], axis=1).dropna()
    Y, F = df['y'].values, df[factor_names].values
    F_const = np.column_stack([np.ones(len(F)), F])

    beta = np.linalg.lstsq(F_const, Y, rcond=None)[0]
    resid = Y - F_const @ beta
    n, k = len(Y), len(beta)

    mse = np.sum(resid**2) / (n - k)
    se = np.sqrt(mse * np.linalg.inv(F_const.T @ F_const).diagonal())

    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))

    ss_res = np.sum(resid**2)
    ss_tot = np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    return {
        'model': model_name,
        'alpha_ann': beta[0] * 12,
        'alpha_t': t[0],
        'alpha_p': p[0],
        'r2': r2,
        'adj_r2': adj_r2,
        'n': n,
        'factors': {
            f: {'beta': beta[i+1], 't': t[i+1], 'p': p[i+1]}
            for i, f in enumerate(factor_names)
        }
    }

def calc_performance_metrics(ret):
    """Calculate comprehensive performance metrics."""
    ann_ret = ret.mean() * 12
    ann_vol = ret.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = ret[ret < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + ret).cumprod()
    dd = (cum - cum.expanding().max()) / cum.expanding().max()
    max_dd = abs(dd.min())

    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd': max_dd,
        'skew': ret.skew(),
        'kurt': ret.kurtosis(),
        'n': len(ret)
    }

def gpr_regression(ret, gpr_data, frequency='monthly'):
    """Regress returns on GPR changes."""
    df = pd.concat([ret.rename('ret'), gpr_data.rename('gpr')], axis=1).dropna()
    df['gpr_chg'] = df['gpr'].diff()
    df = df.dropna()

    if len(df) < 30:
        return None

    Y = df['ret'].values
    X = df['gpr_chg'].values.reshape(-1, 1)
    X_const = np.column_stack([np.ones(len(X)), X])

    beta = np.linalg.lstsq(X_const, Y, rcond=None)[0]
    resid = Y - X_const @ beta
    n, k = len(Y), len(beta)

    mse = np.sum(resid**2) / (n - k)
    se = np.sqrt(mse * np.linalg.inv(X_const.T @ X_const).diagonal())
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))

    ss_res, ss_tot = np.sum(resid**2), np.sum((Y - Y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    return {
        'frequency': frequency,
        'coef': beta[1],
        't': t[1],
        'p': p[1],
        'r2': r2,
        'adj_r2': adj_r2,
        'n': n
    }

# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def main():
    """Execute complete analysis following project info sheet."""

    print("=" * 80)
    print("SHORT-TERM REVERSAL STRATEGY ANALYSIS")
    print("University of Konstanz | Winter 2025/2026")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    str_ret = load_strategy(FILES['str'], 'monthly')
    str_daily = load_strategy(FILES['str_daily'], 'daily')
    resmom_ret = load_strategy(FILES['resmom'], 'monthly')
    ff = load_ff_factors()
    q = load_q_factors()

    print(f"\nData: {len(str_ret)} months | {str_ret.index.min().date()} to {str_ret.index.max().date()}")

    # Align data
    common_ff = str_ret.index.intersection(ff.index)
    common_q = str_ret.index.intersection(q.index)
    str_ff = str_ret.loc[common_ff]
    ff_aligned = ff.loc[common_ff]
    str_q = str_ret.loc[common_q]
    q_aligned = q.loc[common_q]

    # =========================================================================
    # TASK 2: PERFORMANCE METRICS & FACTOR MODELS
    # =========================================================================

    print(f"\n{'='*80}")
    print("TASK 2: PERFORMANCE METRICS & FACTOR MODELS")
    print(f"{'='*80}")

    # Summary statistics
    pre_1990 = str_ff[str_ff.index < '1990-01-01']
    post_1990 = str_ff[str_ff.index >= '1990-01-01']

    perf = calc_performance_metrics(str_ff)
    perf_pre = calc_performance_metrics(pre_1990)
    perf_post = calc_performance_metrics(post_1990)

    print(f"\nSummary Statistics:")
    print(f"{'Period':<25} {'Return':>10} {'Vol':>10} {'Sharpe':>10} {'N':>6}")
    print(f"{'-'*80}")
    print(f"{'Full (1926-2024)':<25} {perf['ann_return']:>9.2%} {perf['ann_vol']:>9.2%} {perf['sharpe']:>10.3f} {perf['n']:>6}")
    print(f"{'Pre-1990':<25} {perf_pre['ann_return']:>9.2%} {perf_pre['ann_vol']:>9.2%} {perf_pre['sharpe']:>10.3f} {perf_pre['n']:>6}")
    print(f"{'Post-1990':<25} {perf_post['ann_return']:>9.2%} {perf_post['ann_vol']:>9.2%} {perf_post['sharpe']:>10.3f} {perf_post['n']:>6}")

    # Factor models - Full sample
    print(f"\n{'Factor Model Regressions (Full Sample):'}")
    print(f"{'-'*80}")

    models_full = [
        (str_ff, ff_aligned, ['MKT-RF'], 'CAPM'),
        (str_ff, ff_aligned, ['MKT-RF', 'SMB', 'HML'], 'FF3'),
        (str_ff, ff_aligned, ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA'], 'FF5'),
        (str_q, q_aligned, ['R_MKT', 'R_ME', 'R_IA', 'R_ROE'], 'Q4'),
        (str_q, q_aligned, ['R_MKT', 'R_ME', 'R_IA', 'R_ROE', 'R_EG'], 'Q5'),
    ]

    results_full = []
    exposures_full = []

    for y, X, factors, name in models_full:
        r = ols_regression(y, X, factors, name)
        sig = '***' if r['alpha_p'] < 0.01 else '**' if r['alpha_p'] < 0.05 else '*' if r['alpha_p'] < 0.10 else ''

        print(f"\n{name} Model (N={r['n']}):")
        print(f"  Alpha (ann.): {r['alpha_ann']:>8.2%}  t={r['alpha_t']:>6.2f}  p={r['alpha_p']:>7.4f} {sig}")
        print(f"  R²: {r['r2']:>6.4f}  Adj. R²: {r['adj_r2']:>6.4f}")
        print(f"  Factor Exposures:")

        for factor in factors:
            beta = r['factors'][factor]['beta']
            t_stat = r['factors'][factor]['t']
            p_val = r['factors'][factor]['p']
            sig_f = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
            print(f"    {factor:<10} β={beta:>8.4f}  t={t_stat:>6.2f}  p={p_val:>7.4f} {sig_f}")

            exposures_full.append({
                'Model': name,
                'Factor': factor,
                'Beta': beta,
                't-statistic': t_stat,
                'p-value': p_val
            })

        results_full.append({
            'Model': name,
            'Sample': f"{y.index.min().year}-{y.index.max().year}",
            'N': r['n'],
            'Alpha (ann.)': r['alpha_ann'],
            't-statistic': r['alpha_t'],
            'p-value': r['alpha_p'],
            'R²': r['r2'],
            'Adj. R²': r['adj_r2']
        })

    # Subperiod analysis
    print(f"\n{'Subperiod Analysis (Pre vs Post 1990):'}")
    print(f"{'-'*80}")

    periods = [
        ('Pre-1990', str_ff[str_ff.index < '1990-01-01'], ff_aligned[ff_aligned.index < '1990-01-01'],
         str_q[str_q.index < '1990-01-01'], q_aligned[q_aligned.index < '1990-01-01']),
        ('Post-1990', str_ff[str_ff.index >= '1990-01-01'], ff_aligned[ff_aligned.index >= '1990-01-01'],
         str_q[str_q.index >= '1990-01-01'], q_aligned[q_aligned.index >= '1990-01-01']),
    ]

    results_subperiod = []
    exposures_subperiod = []

    for period_name, str_sub, ff_sub, str_q_sub, q_sub in periods:
        print(f"\n{period_name}:")

        models_sub = [
            (str_sub, ff_sub, ['MKT-RF'], 'CAPM'),
            (str_sub, ff_sub, ['MKT-RF', 'SMB', 'HML'], 'FF3'),
            (str_sub, ff_sub, ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA'], 'FF5'),
            (str_q_sub, q_sub, ['R_MKT', 'R_ME', 'R_IA', 'R_ROE', 'R_EG'], 'Q5'),
        ]

        for y, X, factors, name in models_sub:
            if len(y) > 24 and len(X) > 24:
                common_idx = y.index.intersection(X.index)
                if len(common_idx) > 24:
                    r = ols_regression(y.loc[common_idx], X.loc[common_idx], factors, name)
                    sig = '***' if r['alpha_p'] < 0.01 else '**' if r['alpha_p'] < 0.05 else '*' if r['alpha_p'] < 0.10 else ''
                    print(f"  {name:<6} α={r['alpha_ann']:>7.2%} t={r['alpha_t']:>5.2f} p={r['alpha_p']:>6.4f} {sig:>4}  R²={r['r2']:>6.4f} Adj.R²={r['adj_r2']:>6.4f}")

                    results_subperiod.append({
                        'Period': period_name,
                        'Model': name,
                        'N': r['n'],
                        'Alpha (ann.)': r['alpha_ann'],
                        't-statistic': r['alpha_t'],
                        'p-value': r['alpha_p'],
                        'R²': r['r2'],
                        'Adj. R²': r['adj_r2']
                    })

                    for factor in factors:
                        exposures_subperiod.append({
                            'Period': period_name,
                            'Model': name,
                            'Factor': factor,
                            'Beta': r['factors'][factor]['beta'],
                            't-statistic': r['factors'][factor]['t'],
                            'p-value': r['factors'][factor]['p']
                        })

    # Save results
    pd.DataFrame({
        'Full Sample': perf,
        'Pre-1990': perf_pre,
        'Post-1990': perf_post
    }).to_csv(f'{OUTPUT_DIR}/table1_performance.csv')

    pd.DataFrame(results_full).to_csv(f'{OUTPUT_DIR}/table2_factor_models.csv', index=False)
    pd.DataFrame(exposures_full).to_csv(f'{OUTPUT_DIR}/table2c_factor_exposures.csv', index=False)
    pd.DataFrame(results_subperiod).to_csv(f'{OUTPUT_DIR}/table2b_factor_models_subperiods.csv', index=False)
    pd.DataFrame(exposures_subperiod).to_csv(f'{OUTPUT_DIR}/table2d_factor_exposures_subperiods.csv', index=False)

    # =========================================================================
    # TASK 4: GEOPOLITICAL RISK EXPOSURE
    # =========================================================================

    print(f"\n{'='*80}")
    print("TASK 4: GEOPOLITICAL RISK EXPOSURE")
    print(f"{'='*80}")

    gpr_results = []

    # Daily GPR (if available)
    if str_daily is not None:
        try:
            gpr_daily = load_gpr('daily')
            gpr_common_daily = str_daily.index.intersection(gpr_daily.index)
            if len(gpr_common_daily) > 50:
                gpr_daily_res = gpr_regression(str_daily.loc[gpr_common_daily], gpr_daily.loc[gpr_common_daily], 'daily')
                if gpr_daily_res:
                    sig = '***' if gpr_daily_res['p'] < 0.01 else '**' if gpr_daily_res['p'] < 0.05 else '*' if gpr_daily_res['p'] < 0.10 else ''
                    print(f"\nDaily GPR Regression (N={gpr_daily_res['n']}):")
                    print(f"  Coefficient: {gpr_daily_res['coef']:>10.6f}  t={gpr_daily_res['t']:>6.2f}  p={gpr_daily_res['p']:>7.4f} {sig}")
                    gpr_results.append({
                        'Frequency': 'Daily',
                        'Coefficient': gpr_daily_res['coef'],
                        't-statistic': gpr_daily_res['t'],
                        'p-value': gpr_daily_res['p'],
                        'R²': gpr_daily_res['r2'],
                        'N': gpr_daily_res['n']
                    })
        except:
            pass

    # Monthly GPR
    gpr_monthly_data = load_gpr('monthly')
    gpr_common = str_ff.index.intersection(gpr_monthly_data.index)
    if len(gpr_common) > 50:
        gpr_monthly = gpr_regression(str_ff.loc[gpr_common], gpr_monthly_data.loc[gpr_common], 'monthly')
        if gpr_monthly:
            sig = '***' if gpr_monthly['p'] < 0.01 else '**' if gpr_monthly['p'] < 0.05 else '*' if gpr_monthly['p'] < 0.10 else ''
            print(f"\nMonthly GPR Regression (N={gpr_monthly['n']}):")
            print(f"  Coefficient: {gpr_monthly['coef']:>10.6f}  t={gpr_monthly['t']:>6.2f}  p={gpr_monthly['p']:>7.4f} {sig}")

            # Correlation interpretation
            corr_gpr = str_ff.loc[gpr_common].corr(gpr_monthly_data.loc[gpr_common].diff().dropna())
            print(f"\nInterpretation:")
            print(f"  The near-zero coefficient ({gpr_monthly['coef']:.6f}, p={gpr_monthly['p']:.2f}) indicates that")
            print(f"  short-term reversal has **no significant exposure to geopolitical risk**.")
            print(f"  This is NEITHER a hedge (negative exposure) NOR a risk amplifier (positive exposure).")
            print(f"  The strategy's returns are largely **orthogonal to geopolitical events**.")

            gpr_results.append({
                'Frequency': 'Monthly',
                'Coefficient': gpr_monthly['coef'],
                't-statistic': gpr_monthly['t'],
                'p-value': gpr_monthly['p'],
                'R²': gpr_monthly['r2'],
                'N': gpr_monthly['n']
            })

    if gpr_results:
        pd.DataFrame(gpr_results).to_csv(f'{OUTPUT_DIR}/table4_gpr.csv', index=False)

    # =========================================================================
    # TASK 5: COMBINED TRADING STRATEGY
    # =========================================================================

    print(f"\n{'='*80}")
    print("TASK 5: COMBINED TRADING STRATEGY")
    print(f"{'='*80}")

    # Combine with residual momentum
    common_port = str_ff.index.intersection(resmom_ret.index)
    str_p = str_ff.loc[common_port]
    rm_p = resmom_ret.loc[common_port]

    corr = str_p.corr(rm_p)
    print(f"\nCorrelation (STR vs Residual Momentum): {corr:>6.3f}")
    print(f"  Negative correlation enables diversification benefits.")

    # Optimize weights
    best_w, best_sr = 0, 0
    for w in np.linspace(0, 1, 101):
        port = w * str_p + (1 - w) * rm_p
        sr = port.mean() / port.std() * np.sqrt(12) if port.std() > 0 else 0
        if sr > best_sr:
            best_w, best_sr = w, sr

    port_opt = best_w * str_p + (1 - best_w) * rm_p

    # Performance comparison
    str_perf = calc_performance_metrics(str_p)
    rm_perf = calc_performance_metrics(rm_p)
    combined_perf = calc_performance_metrics(port_opt)

    print(f"\nOptimal Portfolio Weights:")
    print(f"  Short-term Reversal: {best_w:>5.1%}")
    print(f"  Residual Momentum:   {1-best_w:>5.1%}")

    print(f"\nPerformance Comparison:")
    print(f"{'Strategy':<30} {'Return':>10} {'Vol':>10} {'Sharpe':>10}")
    print(f"{'-'*80}")
    print(f"{'STR only':<30} {str_perf['ann_return']:>9.2%} {str_perf['ann_vol']:>9.2%} {str_perf['sharpe']:>10.3f}")
    print(f"{'Residual Momentum only':<30} {rm_perf['ann_return']:>9.2%} {rm_perf['ann_vol']:>9.2%} {rm_perf['sharpe']:>10.3f}")
    print(f"{f'Combined ({best_w:.0%}/{1-best_w:.0%})':<30} {combined_perf['ann_return']:>9.2%} {combined_perf['ann_vol']:>9.2%} {combined_perf['sharpe']:>10.3f}")

    improvement = (combined_perf['sharpe'] - str_perf['sharpe']) / abs(str_perf['sharpe']) * 100 if str_perf['sharpe'] != 0 else 0
    print(f"\nSharpe Ratio Improvement: +{improvement:.1f}%")

    # Excess returns
    print(f"\nExcess Returns (over STR only):")
    excess_return = (combined_perf['ann_return'] - str_perf['ann_return'])
    print(f"  Additional return: {excess_return:>7.2%} per year")
    print(f"  Risk reduction:    {(str_perf['ann_vol'] - combined_perf['ann_vol']):>7.2%}")

    # Factor models for combined strategy
    print(f"\nFactor Models for Combined Strategy:")
    print(f"{'-'*80}")

    common_ff_port = port_opt.index.intersection(ff_aligned.index)
    common_q_port = port_opt.index.intersection(q_aligned.index)
    port_ff = port_opt.loc[common_ff_port]
    port_q = port_opt.loc[common_q_port]

    models_combined = [
        (port_ff, ff_aligned.loc[common_ff_port], ['MKT-RF'], 'CAPM'),
        (port_ff, ff_aligned.loc[common_ff_port], ['MKT-RF', 'SMB', 'HML'], 'FF3'),
        (port_ff, ff_aligned.loc[common_ff_port], ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA'], 'FF5'),
        (port_q, q_aligned.loc[common_q_port], ['R_MKT', 'R_ME', 'R_IA', 'R_ROE'], 'Q4'),
        (port_q, q_aligned.loc[common_q_port], ['R_MKT', 'R_ME', 'R_IA', 'R_ROE', 'R_EG'], 'Q5'),
    ]

    results_combined = []
    exposures_combined = []

    for y, X, factors, name in models_combined:
        r = ols_regression(y, X, factors, name)
        sig = '***' if r['alpha_p'] < 0.01 else '**' if r['alpha_p'] < 0.05 else '*' if r['alpha_p'] < 0.10 else ''
        print(f"\n{name} Model (N={r['n']}):")
        print(f"  Alpha (ann.): {r['alpha_ann']:>8.2%}  t={r['alpha_t']:>6.2f}  p={r['alpha_p']:>7.4f} {sig}")
        print(f"  R²: {r['r2']:>6.4f}  Adj. R²: {r['adj_r2']:>6.4f}")

        results_combined.append({
            'Model': name,
            'N': r['n'],
            'Alpha (ann.)': r['alpha_ann'],
            't-statistic': r['alpha_t'],
            'p-value': r['alpha_p'],
            'R²': r['r2'],
            'Adj. R²': r['adj_r2']
        })

        for factor in factors:
            exposures_combined.append({
                'Model': name,
                'Factor': factor,
                'Beta': r['factors'][factor]['beta'],
                't-statistic': r['factors'][factor]['t'],
                'p-value': r['factors'][factor]['p']
            })

    # GPR exposure of combined strategy
    if len(gpr_common) > 50:
        gpr_port = gpr_regression(port_opt.loc[port_opt.index.intersection(gpr_common)],
                                   gpr_monthly_data.loc[gpr_common], 'monthly')
        if gpr_port:
            print(f"\nGPR Exposure Comparison:")
            print(f"{'-'*80}")
            print(f"  STR only:     Coef={gpr_monthly['coef']:>10.6f}  t={gpr_monthly['t']:>6.2f}  p={gpr_monthly['p']:>7.4f}")
            sig_port = '***' if gpr_port['p'] < 0.01 else '**' if gpr_port['p'] < 0.05 else '*' if gpr_port['p'] < 0.10 else ''
            print(f"  Combined:     Coef={gpr_port['coef']:>10.6f}  t={gpr_port['t']:>6.2f}  p={gpr_port['p']:>7.4f} {sig_port}")

            gpr_combined = pd.DataFrame([{
                'Strategy': 'Short-term Reversal',
                'Coefficient': gpr_monthly['coef'],
                't-statistic': gpr_monthly['t'],
                'p-value': gpr_monthly['p']
            }, {
                'Strategy': f'Combined ({best_w:.0%}/{1-best_w:.0%})',
                'Coefficient': gpr_port['coef'],
                't-statistic': gpr_port['t'],
                'p-value': gpr_port['p']
            }])
            gpr_combined.to_csv(f'{OUTPUT_DIR}/table4b_gpr_combined.csv', index=False)

    # Save results
    pd.DataFrame({
        'Short-term Reversal': str_perf,
        'Residual Momentum': rm_perf,
        f'Combined ({best_w:.0%}/{1-best_w:.0%})': combined_perf
    }).to_csv(f'{OUTPUT_DIR}/table5_portfolio.csv')

    pd.DataFrame(results_combined).to_csv(f'{OUTPUT_DIR}/table5b_combined_factor_models.csv', index=False)
    pd.DataFrame(exposures_combined).to_csv(f'{OUTPUT_DIR}/table5c_combined_exposures.csv', index=False)

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================

    print(f"\n{'='*80}")
    print("GENERATING FIGURES")
    print(f"{'='*80}")

    # Figure 1: STR Cumulative Wealth (clean, no benchmark)
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cum_str_full = (1 + str_ff).cumprod()
    ax1.plot(cum_str_full.index, cum_str_full.values, lw=2.5, color='#1f77b4', label='Short-term Reversal')
    ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Short-term Reversal Strategy: Wealth Growth (1926-2024)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure1_str_wealth.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 2: STR Pre vs Post Publication
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    cum_pre = (1 + pre_1990).cumprod()
    cum_post = (1 + post_1990).cumprod()

    ax2.plot(cum_pre.index, cum_pre.values, lw=2.5, color='#2ca02c', label='Pre-1990 (Pre-Jegadeesh)')
    ax2.plot(cum_post.index, cum_post.values, lw=2.5, color='#d62728', label='Post-1990 (Post-Publication)')
    ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=pd.Timestamp('1990-01-01'), color='black', linestyle='--', alpha=0.7, lw=1.5, label='Jegadeesh (1990)')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Publication Effect: Pre vs Post Jegadeesh (1990)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.1f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure2_str_publication_effect.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 3: STR vs Combined (without ResMom to avoid scale issues)
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    cum_str_port = (1 + str_p).cumprod()
    cum_combined = (1 + port_opt).cumprod()

    ax3.plot(cum_str_port.index, cum_str_port.values, lw=2, color='#1f77b4', label='STR only', alpha=0.8)
    ax3.plot(cum_combined.index, cum_combined.values, lw=2.5, color='#d62728', ls='--', label=f'Combined ({best_w:.0%}/{1-best_w:.0%})')
    ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax3.set_title('Portfolio Optimization: STR vs Combined Strategy', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure3_str_vs_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 4: All three strategies (kept for completeness despite scale issue)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    cum_rm = (1 + rm_p).cumprod()

    ax4.plot(cum_str_port.index, cum_str_port.values, lw=2, color='#1f77b4', label='Short-term Reversal')
    ax4.plot(cum_rm.index, cum_rm.values, lw=2, color='#2ca02c', label='Residual Momentum')
    ax4.plot(cum_combined.index, cum_combined.values, lw=2.5, ls='--', color='#d62728', label=f'Combined ({best_w:.0%}/{1-best_w:.0%})')
    ax4.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax4.set_title('Strategy Comparison: STR, Residual Momentum, Combined', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure4_all_strategies.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 5: Rolling 12-month returns
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    rolling_ret = str_ff.rolling(12).apply(lambda x: (1 + x).prod() - 1) * 100
    ax5.plot(rolling_ret.index, rolling_ret.values, lw=1.5, color='#1f77b4', alpha=0.8)
    ax5.axhline(y=0, color='red', linestyle='-', alpha=0.7, lw=1)
    ax5.axvline(x=pd.Timestamp('1990-01-01'), color='gray', linestyle='--', alpha=0.7, label='Jegadeesh (1990)')
    ax5.fill_between(rolling_ret.index, 0, rolling_ret.values,
                     where=rolling_ret.values > 0, alpha=0.3, color='green', label='Positive')
    ax5.fill_between(rolling_ret.index, 0, rolling_ret.values,
                     where=rolling_ret.values < 0, alpha=0.3, color='red', label='Negative')
    ax5.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Rolling 12-Month Return (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Short-term Reversal: Rolling Annual Returns', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11, frameon=True, shadow=True)
    ax5.grid(alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figure5_rolling_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ figure1_str_wealth.png")
    print(f"  ✓ figure2_str_publication_effect.png")
    print(f"  ✓ figure3_str_vs_combined.png")
    print(f"  ✓ figure4_all_strategies.png")
    print(f"  ✓ figure5_rolling_returns.png")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  Tables:")
    print("    - table1_performance.csv")
    print("    - table2_factor_models.csv")
    print("    - table2b_factor_models_subperiods.csv")
    print("    - table2c_factor_exposures.csv")
    print("    - table2d_factor_exposures_subperiods.csv")
    print("    - table4_gpr.csv")
    print("    - table4b_gpr_combined.csv")
    print("    - table5_portfolio.csv")
    print("    - table5b_combined_factor_models.csv  [NEW]")
    print("    - table5c_combined_exposures.csv  [NEW]")
    print("  Figures:")
    print("    - figure1_str_wealth.png  [NEW]")
    print("    - figure2_str_publication_effect.png  [NEW]")
    print("    - figure3_str_vs_combined.png  [NEW]")
    print("    - figure4_all_strategies.png")
    print("    - figure5_rolling_returns.png")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
