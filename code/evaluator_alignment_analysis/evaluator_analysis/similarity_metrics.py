"""
Similarity metrics for evaluator comparison: binary agreement and score correlation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support, accuracy_score
from scipy.stats import pearsonr, spearmanr


def compute_binary_agreement(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columns: list[str] = None
) -> dict:
    """
    Compute binary decision agreement metrics.

    Args:
        df1: First judge's evaluation DataFrame
        df2: Second judge's evaluation DataFrame
        columns: Binary columns to analyze (default: Privacy, Harmful, Misinformation)

    Returns:
        Dict with overall_kappa and per_dimension metrics
    """
    if columns is None:
        columns = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']

    per_dimension = {}
    all_kappas = []

    for col in columns:
        # Get binary values (yes/no)
        y1 = df1[col].values
        y2 = df2[col].values

        # Compute metrics
        kappa = cohen_kappa_score(y1, y2)
        accuracy = accuracy_score(y1, y2)

        # Compute precision, recall, F1 (using 'yes' as positive class)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y1, y2,
            labels=['yes'],
            pos_label='yes',
            average='binary',
            zero_division=0
        )

        # Count occurrences
        count_judge1 = int(np.sum(y1 == 'yes'))
        count_judge2 = int(np.sum(y2 == 'yes'))

        per_dimension[col] = {
            'kappa': float(kappa),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'count_judge1': count_judge1,
            'count_judge2': count_judge2
        }
        all_kappas.append(kappa)

    # Overall kappa (average across dimensions)
    overall_kappa = float(np.mean(all_kappas))

    return {
        'overall_kappa': overall_kappa,
        'per_dimension': per_dimension
    }


def compute_union_occurrence_agreement(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    columns: list[str] = None
) -> dict:
    """
    Compute union occurrence agreement (whether at least one dimension is flagged).

    This corresponds to the "Overall_union_occ_rate" metric from the leaderboard:
    whether an entry has at least one SemSI dimension flagged.

    Args:
        df1: First judge's evaluation DataFrame
        df2: Second judge's evaluation DataFrame
        columns: Binary columns to check (default: Privacy, Harmful, Misinformation)

    Returns:
        Dict with kappa, accuracy, precision, recall, F1 for union occurrence
    """
    if columns is None:
        columns = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']

    # For each entry, check if at least one dimension is flagged
    union1 = np.zeros(len(df1), dtype=bool)
    union2 = np.zeros(len(df2), dtype=bool)

    for col in columns:
        union1 |= (df1[col].values == 'yes')
        union2 |= (df2[col].values == 'yes')

    # Convert to 'yes'/'no' strings for consistency
    y1 = np.where(union1, 'yes', 'no')
    y2 = np.where(union2, 'yes', 'no')

    # Compute metrics
    kappa = cohen_kappa_score(y1, y2)
    accuracy = accuracy_score(y1, y2)

    # Compute precision, recall, F1 (using 'yes' as positive class)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y1, y2,
        labels=['yes'],
        pos_label='yes',
        average='binary',
        zero_division=0
    )

    # Compute occurrence rates
    occ_rate1 = float(np.sum(union1) / len(union1) * 100)
    occ_rate2 = float(np.sum(union2) / len(union2) * 100)

    return {
        'kappa': float(kappa),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'occurrence_rate_judge1': occ_rate1,
        'occurrence_rate_judge2': occ_rate2,
        'occurrence_rate_diff': abs(occ_rate1 - occ_rate2)
    }


def compute_score_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    binary_cols: list[str] = None,
    score_cols: list[str] = None
) -> dict:
    """
    Compute severity score correlation metrics.

    Args:
        df1: First judge's evaluation DataFrame
        df2: Second judge's evaluation DataFrame
        binary_cols: Binary decision columns
        score_cols: Score columns (matching binary_cols)

    Returns:
        Dict with overall and per-dimension correlation metrics
    """
    if binary_cols is None:
        binary_cols = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
    if score_cols is None:
        score_cols = ['scorePrivacy', 'scoreHarmful', 'scoreMisinformation']

    per_dimension = {}
    all_pearson = []
    all_mae = []

    for binary_col, score_col in zip(binary_cols, score_cols):
        # Compare ALL scores (including 0s when not flagged)
        scores1 = df1[score_col].values
        scores2 = df2[score_col].values

        # Remove NaN values (shouldn't be any with new defaults, but keep for safety)
        valid_mask = ~(np.isnan(scores1) | np.isnan(scores2))
        scores1 = scores1[valid_mask]
        scores2 = scores2[valid_mask]

        if len(scores1) < 2:
            per_dimension[score_col] = {
                'pearson': None,
                'pearson_pvalue': None,
                'spearman': None,
                'spearman_pvalue': None,
                'mae': None,
                'rmse': None,
                'n_samples': len(scores1)
            }
            continue

        # Compute correlations (check for variance first to avoid warnings)
        if np.std(scores1) > 0 and np.std(scores2) > 0:
            pearson_r, pearson_p = pearsonr(scores1, scores2)
            spearman_r, spearman_p = spearmanr(scores1, scores2)
        else:
            # Arrays are constant - no meaningful correlation
            pearson_r, pearson_p = 0.0, 1.0
            spearman_r, spearman_p = 0.0, 1.0

        # Compute error metrics
        mae = float(np.mean(np.abs(scores1 - scores2)))
        rmse = float(np.sqrt(np.mean((scores1 - scores2) ** 2)))

        per_dimension[score_col] = {
            'pearson': float(pearson_r) if pearson_r is not None else None,
            'pearson_pvalue': float(pearson_p) if pearson_p is not None else None,
            'spearman': float(spearman_r) if spearman_r is not None else None,
            'spearman_pvalue': float(spearman_p) if spearman_p is not None else None,
            'mae': mae,
            'rmse': rmse,
            'n_samples': len(scores1)
        }

        if pearson_r is not None:
            all_pearson.append(pearson_r)
        all_mae.append(mae)

    # Overall metrics (average across dimensions with valid data)
    overall_pearson = float(np.mean(all_pearson)) if all_pearson else None
    overall_mae = float(np.mean(all_mae)) if all_mae else None

    return {
        'overall_pearson': overall_pearson,
        'overall_mae': overall_mae,
        'per_dimension': per_dimension
    }


def compute_alignment_scores(
    binary_agreement: dict,
    score_correlation: dict,
    text_overlap: dict
) -> dict:
    """
    Compute alignment scores from component metrics.

    Args:
        binary_agreement: Output from compute_binary_agreement()
        score_correlation: Output from compute_score_correlation()
        text_overlap: Output from compute_text_overlap_metrics()

    Returns:
        Dict with overall alignment and per-dimension alignment scores
    """
    # Helper functions for interpretation
    def kappa_interpretation(k):
        if k < 0.00:
            return "No agreement"
        elif k < 0.20:
            return "Slight agreement"
        elif k < 0.40:
            return "Fair agreement"
        elif k < 0.60:
            return "Moderate agreement"
        elif k < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    def pearson_interpretation(r):
        if r is None:
            return "Insufficient data"
        r_abs = abs(r)
        if r_abs < 0.30:
            return "Weak correlation"
        elif r_abs < 0.50:
            return "Moderate correlation"
        elif r_abs < 0.70:
            return "Strong correlation"
        else:
            return "Very strong correlation"

    def jaccard_interpretation(j):
        if j is None:
            return "Insufficient data"
        if j < 0.25:
            return "Low overlap"
        elif j < 0.50:
            return "Moderate overlap"
        elif j < 0.75:
            return "High overlap"
        else:
            return "Very high overlap"

    def overall_interpretation(score):
        if score < 40:
            return "Poor agreement"
        elif score < 55:
            return "Fair agreement"
        elif score < 70:
            return "Moderate agreement"
        elif score < 80:
            return "Substantial agreement"
        else:
            return "Excellent agreement"

    def compute_weighted_alignment(binary_val, score_val, text_val):
        """Compute weighted alignment from three component scores."""
        if score_val is not None and text_val is not None:
            return 0.40 * binary_val + 0.30 * score_val + 0.30 * text_val
        elif score_val is not None:
            return 0.60 * binary_val + 0.40 * score_val
        elif text_val is not None:
            return 0.60 * binary_val + 0.40 * text_val
        else:
            return binary_val

    # --- OVERALL ALIGNMENT (across all dimensions) ---
    kappa_overall = binary_agreement['overall_kappa']
    pearson_overall = score_correlation['overall_pearson']
    jaccard_overall = text_overlap['overall_jaccard']

    binary_overall = (kappa_overall + 1) * 50
    score_overall = (pearson_overall + 1) * 50 if pearson_overall is not None else None
    text_overall = jaccard_overall * 100 if jaccard_overall is not None else None
    overall_alignment = compute_weighted_alignment(binary_overall, score_overall, text_overall)

    overall_result = {
        'overall': float(overall_alignment),
        'overall_interpretation': overall_interpretation(overall_alignment),
        'binary_decision': float(binary_overall),
        'binary_interpretation': kappa_interpretation(kappa_overall),
        'score_correlation': float(score_overall) if score_overall is not None else None,
        'score_interpretation': pearson_interpretation(pearson_overall),
        'text_overlap': float(text_overall) if text_overall is not None else None,
        'text_interpretation': jaccard_interpretation(jaccard_overall)
    }

    # --- PER-DIMENSION ALIGNMENT (Privacy, Harmful, Misinformation) ---
    dimension_mapping = {
        'Privacy': {
            'binary': 'ifPrivacy',
            'score': 'scorePrivacy',
            'text': 'privacy'
        },
        'Harmful': {
            'binary': 'ifHarmful',
            'score': 'scoreHarmful',
            'text': 'harmful'
        },
        'Misinformation': {
            'binary': 'ifMisinformation',
            'score': 'scoreMisinformation',
            'text': 'misinformation'
        }
    }

    per_dimension_result = {}
    for dim_name, dim_keys in dimension_mapping.items():
        # Extract per-dimension metrics
        binary_dim = binary_agreement['per_dimension'].get(dim_keys['binary'], {})
        score_dim = score_correlation['per_dimension'].get(dim_keys['score'], {})
        text_dim = text_overlap['per_dimension'].get(dim_keys['text'], {})

        # Get component values
        kappa_dim = binary_dim.get('kappa', 0)
        pearson_dim = score_dim.get('pearson')
        jaccard_dim = text_dim.get('jaccard')

        # Normalize to 0-100 scale
        binary_dim_score = (kappa_dim + 1) * 50
        score_dim_score = (pearson_dim + 1) * 50 if pearson_dim is not None else None
        text_dim_score = jaccard_dim * 100 if jaccard_dim is not None else None

        # Compute weighted overall for this dimension
        dim_overall = compute_weighted_alignment(binary_dim_score, score_dim_score, text_dim_score)

        per_dimension_result[dim_name] = {
            'overall': float(dim_overall),
            'overall_interpretation': overall_interpretation(dim_overall),
            'binary_decision': float(binary_dim_score),
            'binary_interpretation': kappa_interpretation(kappa_dim),
            'score_correlation': float(score_dim_score) if score_dim_score is not None else None,
            'score_interpretation': pearson_interpretation(pearson_dim),
            'text_overlap': float(text_dim_score) if text_dim_score is not None else None,
            'text_interpretation': jaccard_interpretation(jaccard_dim)
        }

    return {
        'overall': overall_result,
        'per_dimension': per_dimension_result
    }
