"""
Text overlap metrics for evaluator comparison using bracketed text analysis.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def extract_marked_positions(text: str, reference: str) -> list[tuple[int, int]]:
    """
    Extract positions of <<<marked>>> spans in text.

    Args:
        text: Text with <<<marked>>> spans
        reference: Reference text (unmarked)

    Returns:
        List of (start, end) position tuples in reference text
    """
    if not isinstance(text, str) or not isinstance(reference, str):
        return []

    pattern = r'<<<(.*?)>>>'
    matches = re.finditer(pattern, text, re.DOTALL)
    positions = []

    for match in matches:
        marked_content = match.group(1)
        # Find this content in the reference text
        start_ref = reference.find(marked_content)
        if start_ref != -1:
            end_ref = start_ref + len(marked_content)
            positions.append((start_ref, end_ref))

    return positions


def compute_jaccard_similarity(
    positions1: list[tuple[int, int]],
    positions2: list[tuple[int, int]],
    text_length: int
) -> float:
    """
    Compute Jaccard similarity between two sets of marked positions.

    Args:
        positions1: First set of (start, end) tuples
        positions2: Second set of (start, end) tuples
        text_length: Total length of reference text

    Returns:
        Jaccard similarity (0-1)
    """
    if text_length == 0:
        return 0.0

    # Create boolean arrays for marked positions
    marked1 = [False] * text_length
    marked2 = [False] * text_length

    for start, end in positions1:
        for i in range(start, min(end, text_length)):
            marked1[i] = True

    for start, end in positions2:
        for i in range(start, min(end, text_length)):
            marked2[i] = True

    # Compute intersection and union
    intersection = sum(m1 and m2 for m1, m2 in zip(marked1, marked2))
    union = sum(m1 or m2 for m1, m2 in zip(marked1, marked2))

    if union == 0:
        return 0.0

    return intersection / union


def compute_coverage(text: str) -> float:
    """
    Compute coverage percentage of bracketed text.

    Args:
        text: Text with <<<marked>>> spans

    Returns:
        Coverage percentage (0-1)
    """
    if not isinstance(text, str):
        return 0.0

    bracketed_parts = re.findall(r'<<<(.*?)>>>', text, re.DOTALL)
    bracketed_length = sum(len(part) for part in bracketed_parts)
    total_length = len(text) - text.count('<<<') * 3 - text.count('>>>') * 3

    if total_length == 0:
        return 0.0

    return bracketed_length / total_length


def compute_text_overlap_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    text_columns: list[str] = None,
    answer_column: str = 'answer'
) -> dict:
    """
    Compute text overlap metrics using bracketed text analysis.

    Args:
        df1: First judge's evaluation DataFrame
        df2: Second judge's evaluation DataFrame
        text_columns: Columns with marked text (default: privacy, harmful, misinformation)
        answer_column: Column with reference text

    Returns:
        Dict with overall_jaccard and per_dimension metrics
    """
    if text_columns is None:
        text_columns = ['privacy', 'harmful', 'misinformation']

    per_dimension = {}
    all_jaccard = []

    for col in text_columns:
        # Check if columns exist
        if col not in df1.columns or col not in df2.columns:
            per_dimension[col] = {
                'jaccard': None,
                'coverage_correlation': None,
                'coverage_corr_pvalue': None,
                'n_samples': 0
            }
            continue

        # Compute Jaccard for each record
        jaccards = []
        coverages1 = []
        coverages2 = []

        for idx in range(len(df1)):
            text1 = df1.iloc[idx][col]
            text2 = df2.iloc[idx][col]
            reference = df1.iloc[idx][answer_column]

            # Skip if either text is null
            if not isinstance(text1, str) or not isinstance(text2, str):
                continue
            if not isinstance(reference, str):
                continue

            # Extract positions
            positions1 = extract_marked_positions(text1, reference)
            positions2 = extract_marked_positions(text2, reference)

            # Compute Jaccard for all entries:
            # - Both marked: normal Jaccard intersection/union
            # - One-sided: Jaccard = 0 (empty vs non-empty)
            # - Neither marked: Jaccard = 1 (perfect agreement on nothing)
            if not positions1 and not positions2:
                jaccards.append(1.0)
            else:
                jaccard = compute_jaccard_similarity(positions1, positions2, len(reference))
                jaccards.append(jaccard)

            # Compute coverages
            cov1 = compute_coverage(text1)
            cov2 = compute_coverage(text2)
            coverages1.append(cov1)
            coverages2.append(cov2)

        # Compute metrics
        if jaccards:
            mean_jaccard = float(np.mean(jaccards))
            all_jaccard.append(mean_jaccard)
        else:
            mean_jaccard = 0.0
            all_jaccard.append(0.0)

        # Compute coverage correlation (check for variance first)
        if len(coverages1) >= 2 and len(coverages2) >= 2:
            # Check if either array is constant
            if np.std(coverages1) > 0 and np.std(coverages2) > 0:
                cov_corr, cov_p = pearsonr(coverages1, coverages2)
                cov_corr = float(cov_corr)
                cov_p = float(cov_p)
            else:
                # Constant coverage - no meaningful correlation
                cov_corr = 0.0
                cov_p = 1.0
        else:
            cov_corr = 0.0
            cov_p = 1.0

        per_dimension[col] = {
            'jaccard': mean_jaccard,
            'coverage_correlation': cov_corr,
            'coverage_corr_pvalue': cov_p,
            'n_samples': len(jaccards)
        }

    # Overall Jaccard (average across dimensions)
    overall_jaccard = float(np.mean(all_jaccard)) if all_jaccard else 0.0

    return {
        'overall_jaccard': overall_jaccard,
        'per_dimension': per_dimension
    }
