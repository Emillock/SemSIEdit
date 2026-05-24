"""
Evaluator Similarity Comparison Tool

This module provides functions to measure alignment and disagreement between
two evaluator results in the SemSI project.
"""

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .data_loader import (
    find_shared_model_methods,
    filter_by_min_samples,
    load_evaluator_pair,
    load_internal_evaluations,
    _find_repo_root
)

__all__ = [
    'compare_evaluators',
    'find_shared_model_methods',
    'filter_by_min_samples',
    'load_evaluator_pair'
]

from .similarity_metrics import compute_binary_agreement, compute_score_correlation, compute_alignment_scores, compute_union_occurrence_agreement
from .text_overlap import compute_text_overlap_metrics
from .gap_metrics import compute_gap_metrics
from .report_generator import generate_comparison_report, generate_batch_report, save_metrics_json


def compare_evaluators(
    judge1: str = None,
    judge2: str = None,
    model: str | None = None,
    method: str | None = None,
    output_format: str = 'console',
    save_json: str | None = None,
    min_samples: int = 99,
    mode: str = 'judge'
):
    """
    Compare two evaluators and return comprehensive metrics.

    Args:
        judge1: First judge name (e.g., 'gpt-5'). In evaluator mode, this is the judge to compare against.
        judge2: Second judge name (e.g., 'gpt-oss-20b'). Only used in judge mode.
        model: Model name (optional in judge mode - if None, compare all shared models. Required in evaluator mode.)
        method: Method name (optional in judge mode - if None, compare all shared methods. Required in evaluator mode.)
        output_format: Output format ('console' | 'json' | 'dict')
        save_json: Optional path to save JSON output
        min_samples: Minimum samples required per dataset (default 99)
        mode: Comparison mode ('judge' | 'evaluator').
              'judge': Compare two external judges (default).
              'evaluator': Compare model's internal evaluator vs external judge.

    Returns:
        dict or list[dict]: Metrics for single or multiple comparisons
    """
    # Mode routing
    if mode == 'evaluator':
        # Evaluator mode: Compare internal evaluator vs judge
        if model is None or method is None:
            raise ValueError("Evaluator mode requires both 'model' and 'method' parameters")
        if 'Feedback' not in method or ('Iter' not in method and 'OnlyPre' not in method):
            raise ValueError("Evaluator mode only works with Feedback(NIter) or Feedback(OnlyPre) datasets")

        judge = judge1 if judge1 is not None else 'gpt-5'
        result = _compare_evaluator_mode(model, method, judge)

        # Format output
        if output_format == 'console':
            report = generate_comparison_report(
                f'{model} (internal)', judge, model, method, result, 'console'
            )
            print(report)
        elif output_format == 'json':
            report = generate_comparison_report(
                f'{model} (internal)', judge, model, method, result, 'json'
            )
            print(report)

        if save_json:
            save_metrics_json(result, save_json)

        if output_format == 'dict':
            return result
        else:
            return report

    # Judge mode (existing behavior)
    elif mode == 'judge':
        if judge1 is None or judge2 is None:
            raise ValueError("Judge mode requires both 'judge1' and 'judge2' parameters")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'judge' or 'evaluator'")

    # Case 1: Specific model and method provided
    if model is not None and method is not None:
        result = _compare_single(judge1, judge2, model, method)

        if output_format == 'console':
            report = generate_comparison_report(judge1, judge2, model, method, result, 'console')
            print(report)
        elif output_format == 'json':
            report = generate_comparison_report(judge1, judge2, model, method, result, 'json')
            print(report)

        if save_json:
            save_metrics_json(result, save_json)

        if output_format == 'dict':
            return result
        else:
            return report

    # Case 2: Batch comparison - all shared model/method pairs
    else:
        # Find shared pairs
        shared_pairs = find_shared_model_methods(judge1, judge2)

        # Filter by minimum samples
        valid_pairs, skipped_info = filter_by_min_samples(
            judge1, judge2, shared_pairs, min_samples
        )

        # Compare each pair
        all_results = []

        # Set up progress bar
        if HAS_TQDM:
            pairs_iter = tqdm(valid_pairs, desc="Comparing evaluators", unit="pair")
        else:
            pairs_iter = valid_pairs
            print(f"Comparing {len(valid_pairs)} model/method pairs...")

        for idx, (m, meth) in enumerate(pairs_iter, 1):
            try:
                result = _compare_single(judge1, judge2, m, meth)
                all_results.append(result)

                # Simple progress for non-tqdm case
                if not HAS_TQDM:
                    print(f"  [{idx}/{len(valid_pairs)}] Completed: {m} - {meth}")
            except Exception as e:
                print(f"Warning: Failed to compare {m}-{meth}: {str(e)}")

        if output_format == 'console':
            report = generate_batch_report(judge1, judge2, all_results, skipped_info, 'console')
            print(report)
        elif output_format == 'json':
            report = generate_batch_report(judge1, judge2, all_results, skipped_info, 'json')
            print(report)

        if save_json:
            save_metrics_json(all_results, save_json)

        if output_format == 'dict':
            return all_results
        else:
            return report


def _compare_single(judge1: str, judge2: str, model: str, method: str) -> dict:
    """Compare a single model/method pair between two judges."""
    # Load data
    df1, df2, sample_size = load_evaluator_pair(judge1, judge2, model, method)

    # Compute metrics
    binary_agreement = compute_binary_agreement(df1, df2)
    score_correlation = compute_score_correlation(df1, df2)
    text_overlap = compute_text_overlap_metrics(df1, df2)
    union_occurrence = compute_union_occurrence_agreement(df1, df2)
    alignment_scores = compute_alignment_scores(binary_agreement, score_correlation, text_overlap)
    gap_metrics = compute_gap_metrics(df1, df2, judge1, judge2)

    # Assemble result
    result = {
        'comparison_metadata': {
            'judge1': judge1,
            'judge2': judge2,
            'model': model,
            'method': method,
            'sample_size': sample_size
        },
        'alignment_scores': alignment_scores,
        'binary_agreement': binary_agreement,
        'score_correlation': score_correlation,
        'text_overlap': text_overlap,
        'union_occurrence': union_occurrence,
        'gap_metrics': gap_metrics
    }

    return result


def _compare_evaluator_mode(model: str, method: str, judge: str) -> dict:
    """Compare internal evaluator vs external judge."""
    import pandas as pd
    from pathlib import Path

    # Load internal evaluations from logs
    df_internal = load_internal_evaluations(model, method)

    # Load judge evaluations
    repo_root = _find_repo_root()

    # Map Feedback(NIter) to Feedback(OnlyPre) for judge comparison
    # This compares first iteration internal eval against OnlyPre judge
    if 'FeedbackFullEval' in method and 'Iter' in method:
        judge_method = 'FeedbackFullEval(OnlyPre)'
    elif 'Feedback' in method and 'Iter' in method:
        judge_method = 'Feedback(OnlyPre)'
    else:
        judge_method = method
    judge_path = repo_root / 'datasets' / 'evaluations' / judge / f'{model}-{judge_method}_label.jsonl'

    if not judge_path.exists():
        raise FileNotFoundError(f"Judge evaluation not found: {judge_path}")

    df_judge = pd.read_json(judge_path, lines=True)

    # Filter valid entries (judge dataset)
    if 'error' in df_judge.columns:
        df_judge = df_judge[
            (df_judge['error'].isna()) |
            (df_judge['error'] == None) |
            (df_judge['error'].apply(lambda x: isinstance(x, float) and pd.isna(x)))
        ].copy()

    # Convert judge binary fields to string
    binary_fields = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
    for field in binary_fields:
        if field in df_judge.columns:
            df_judge[field] = df_judge[field].astype(str).str.lower()

    # Ensure judge score fields exist with default 0
    score_fields = ['scorePrivacy', 'scoreHarmful', 'scoreMisinformation']
    for field in score_fields:
        if field not in df_judge.columns:
            df_judge[field] = 0
        df_judge[field] = pd.to_numeric(df_judge[field], errors='coerce').fillna(0)

    # Ensure judge text fields exist with default ''
    text_fields = ['privacy', 'harmful', 'misinformation']
    for field in text_fields:
        if field not in df_judge.columns:
            df_judge[field] = ''
        df_judge[field] = df_judge[field].fillna('')

    # Align datasets by ID (inner join — only keep entries present in both)
    common_ids = set(df_internal['ID']) & set(df_judge['ID'])
    df_internal = df_internal[df_internal['ID'].isin(common_ids)].sort_values('ID').reset_index(drop=True)
    df_judge = df_judge[df_judge['ID'].isin(common_ids)].sort_values('ID').reset_index(drop=True)
    min_size = len(df_internal)

    # Ensure 'answer' column exists for text overlap
    if 'answer' not in df_internal.columns:
        df_internal['answer'] = ''
    if 'answer' not in df_judge.columns:
        df_judge['answer'] = ''

    # Compute metrics
    binary_agreement = compute_binary_agreement(df_internal, df_judge)
    score_correlation = compute_score_correlation(df_internal, df_judge)
    text_overlap = compute_text_overlap_metrics(df_internal, df_judge, answer_column='answer')
    union_occurrence = compute_union_occurrence_agreement(df_internal, df_judge)
    alignment_scores = compute_alignment_scores(binary_agreement, score_correlation, text_overlap)
    gap_metrics = compute_gap_metrics(df_internal, df_judge, f'{model} (internal)', judge)

    # Assemble result
    result = {
        'comparison_metadata': {
            'judge1': f'{model} (internal evaluator - first iteration)',
            'judge2': judge,
            'model': model,
            'method': method,
            'judge_method': judge_method,
            'sample_size': min_size
        },
        'alignment_scores': alignment_scores,
        'binary_agreement': binary_agreement,
        'score_correlation': score_correlation,
        'text_overlap': text_overlap,
        'union_occurrence': union_occurrence,
        'gap_metrics': gap_metrics
    }

    return result
