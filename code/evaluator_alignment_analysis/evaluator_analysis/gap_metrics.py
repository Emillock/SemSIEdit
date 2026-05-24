"""
Gap metrics for measuring disagreement between evaluators.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from .text_overlap import compute_coverage

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_MCNEMAR = True
except ImportError:
    HAS_MCNEMAR = False


def compute_gap_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    judge1_name: str = "Judge 1",
    judge2_name: str = "Judge 2"
) -> dict:
    """
    Compute gap/disagreement metrics between two evaluators.

    Args:
        df1: First judge's evaluation DataFrame
        df2: Second judge's evaluation DataFrame
        judge1_name: Name of first judge for display
        judge2_name: Name of second judge for display

    Returns:
        Dict with overall_gap, decision_gap, severity_gap, coverage_gap,
        direction_bias, and statistical_tests
    """
    binary_cols = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
    score_cols = ['scorePrivacy', 'scoreHarmful', 'scoreMisinformation']
    text_cols = ['privacy', 'harmful', 'misinformation']
    dimension_names = ['Privacy', 'Harmful', 'Misinformation']

    # Decision Gap (% disagreement on yes/no)
    decision_gap = {}
    for binary_col, dim_name in zip(binary_cols, dimension_names):
        disagreement_pct = float(np.mean(df1[binary_col] != df2[binary_col]) * 100)
        decision_gap[dim_name] = disagreement_pct

    # Severity Gap (MAE across all entries)
    severity_gap = {}
    for binary_col, score_col, dim_name in zip(binary_cols, score_cols, dimension_names):
        scores1 = df1[score_col].values
        scores2 = df2[score_col].values
        valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))
        if valid_mask.sum() > 0:
            mae = float(np.mean(np.abs(scores1[valid_mask] - scores2[valid_mask])))
            severity_gap[dim_name] = mae
        else:
            severity_gap[dim_name] = None

    # Coverage Gap (mean absolute difference in marked text coverage)
    coverage_gap = {}
    for text_col, dim_name in zip(text_cols, dimension_names):
        if text_col in df1.columns and text_col in df2.columns:
            coverages1 = df1[text_col].apply(compute_coverage).values
            coverages2 = df2[text_col].apply(compute_coverage).values
            gap = float(np.mean(np.abs(coverages1 - coverages2)) * 100)
            coverage_gap[dim_name] = gap
        else:
            coverage_gap[dim_name] = None

    # Direction Bias (which judge is stricter)
    # Count cases where judge1 flagged but judge2 didn't, and vice versa
    judge1_stricter = 0
    judge2_stricter = 0
    symmetric = 0

    for binary_col in binary_cols:
        y1 = df1[binary_col].values
        y2 = df2[binary_col].values

        j1_only = np.sum((y1 == 'yes') & (y2 == 'no'))
        j2_only = np.sum((y1 == 'no') & (y2 == 'yes'))
        both = np.sum((y1 == 'yes') & (y2 == 'yes'))
        neither = np.sum((y1 == 'no') & (y2 == 'no'))

        judge1_stricter += j1_only
        judge2_stricter += j2_only
        symmetric += both + neither

    total = judge1_stricter + judge2_stricter + symmetric
    direction_bias = {
        f'{judge1_name}_stricter_pct': float(judge1_stricter / total * 100) if total > 0 else 0,
        f'{judge2_name}_stricter_pct': float(judge2_stricter / total * 100) if total > 0 else 0,
        'symmetric_pct': float(symmetric / total * 100) if total > 0 else 0
    }

    # Statistical Tests
    statistical_tests = {}

    # McNemar test (for binary asymmetry) - use Privacy dimension
    if HAS_MCNEMAR:
        try:
            y1 = (df1['ifPrivacy'] == 'yes').astype(int).values
            y2 = (df2['ifPrivacy'] == 'yes').astype(int).values

            # Create contingency table
            both_yes = np.sum((y1 == 1) & (y2 == 1))
            both_no = np.sum((y1 == 0) & (y2 == 0))
            y1_yes_y2_no = np.sum((y1 == 1) & (y2 == 0))
            y1_no_y2_yes = np.sum((y1 == 0) & (y2 == 1))

            contingency = [[both_yes, y1_yes_y2_no],
                           [y1_no_y2_yes, both_no]]

            if y1_yes_y2_no + y1_no_y2_yes >= 25:  # McNemar requires enough discordant pairs
                result = mcnemar(contingency, exact=False)
                statistical_tests['mcnemar'] = {
                    'statistic': float(result.statistic),
                    'pvalue': float(result.pvalue),
                    'test': 'Binary decision asymmetry (Privacy)'
                }
            else:
                statistical_tests['mcnemar'] = {
                    'statistic': None,
                    'pvalue': None,
                    'test': 'Insufficient discordant pairs for McNemar test'
                }
        except Exception as e:
            statistical_tests['mcnemar'] = {
                'statistic': None,
                'pvalue': None,
                'test': f'Error: {str(e)}'
            }
    else:
        statistical_tests['mcnemar'] = {
            'statistic': None,
            'pvalue': None,
            'test': 'statsmodels not available'
        }

    # Wilcoxon test (for paired score differences) - use Privacy scores
    try:
        mask = (df1['ifPrivacy'] == 'yes') & (df2['ifPrivacy'] == 'yes')
        if mask.sum() >= 3:  # Need at least 3 pairs
            scores1 = df1.loc[mask, 'scorePrivacy'].values
            scores2 = df2.loc[mask, 'scorePrivacy'].values
            valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))

            if valid_mask.sum() >= 3:
                scores1_valid = scores1[valid_mask]
                scores2_valid = scores2[valid_mask]

                # Check if scores differ - Wilcoxon requires variation
                if not np.array_equal(scores1_valid, scores2_valid):
                    result = wilcoxon(scores1_valid, scores2_valid)
                    statistical_tests['wilcoxon'] = {
                        'statistic': float(result.statistic),
                        'pvalue': float(result.pvalue),
                        'test': 'Severity score differences (Privacy)'
                    }
                else:
                    # Identical scores - no differences to test
                    statistical_tests['wilcoxon'] = {
                        'statistic': None,
                        'pvalue': None,
                        'test': 'Identical scores (no variation)'
                    }
            else:
                statistical_tests['wilcoxon'] = {
                    'statistic': None,
                    'pvalue': None,
                    'test': 'Insufficient valid pairs for Wilcoxon test'
                }
        else:
            statistical_tests['wilcoxon'] = {
                'statistic': None,
                'pvalue': None,
                'test': 'Insufficient flagged cases for Wilcoxon test'
            }
    except Exception as e:
        statistical_tests['wilcoxon'] = {
            'statistic': None,
            'pvalue': None,
            'test': f'Error: {str(e)}'
        }

    # Overall gap (average decision gap across dimensions)
    overall_gap = float(np.mean(list(decision_gap.values())))

    return {
        'overall_gap': overall_gap,
        'decision_gap': decision_gap,
        'severity_gap': severity_gap,
        'coverage_gap': coverage_gap,
        'direction_bias': direction_bias,
        'statistical_tests': statistical_tests
    }
