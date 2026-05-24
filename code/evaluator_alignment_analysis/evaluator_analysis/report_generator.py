"""
Report generation for evaluator comparison results.
"""

import json
from typing import Dict, List


def generate_comparison_report(
    judge1: str,
    judge2: str,
    model: str,
    method: str,
    metrics: dict,
    output_format: str = 'console'
) -> str | dict:
    """
    Generate formatted comparison report.

    Args:
        judge1: First judge name
        judge2: Second judge name
        model: Model name
        method: Method name
        metrics: Complete metrics dict
        output_format: 'console' or 'json' or 'dict'

    Returns:
        Formatted string (console/json) or dict
    """
    if output_format == 'dict':
        return metrics
    elif output_format == 'json':
        return json.dumps(metrics, indent=2)
    else:  # console
        return _format_console_report(judge1, judge2, model, method, metrics)


def generate_batch_report(
    judge1: str,
    judge2: str,
    all_metrics: List[dict],
    skipped_info: dict,
    output_format: str = 'console'
) -> str | list:
    """
    Generate report for batch comparison (multiple model/method pairs).

    Args:
        judge1: First judge name
        judge2: Second judge name
        all_metrics: List of metrics dicts
        skipped_info: Dict of skipped pairs with reasons
        output_format: 'console' or 'json' or 'dict'

    Returns:
        Formatted report
    """
    if output_format == 'dict':
        return all_metrics
    elif output_format == 'json':
        return json.dumps(all_metrics, indent=2)
    else:  # console
        return _format_batch_console_report(judge1, judge2, all_metrics, skipped_info)


def _format_console_report(judge1: str, judge2: str, model: str, method: str, metrics: dict) -> str:
    """Format single comparison as console output."""
    lines = []
    meta = metrics['comparison_metadata']
    scores = metrics['alignment_scores']
    binary = metrics['binary_agreement']
    score_corr = metrics['score_correlation']
    text_ov = metrics['text_overlap']
    union_occ = metrics.get('union_occurrence', {})
    gaps = metrics['gap_metrics']
    sample_size = meta['sample_size']

    # Header
    lines.append("=" * 80)
    lines.append("Evaluator Comparison Report")
    lines.append("=" * 80)
    lines.append(f"Judges:        {judge1} vs {judge2}")
    lines.append(f"Model:         {model}")
    lines.append(f"Method:        {method}")
    lines.append(f"Sample Size:   {meta['sample_size']} matched records")
    lines.append("")

    # Alignment Scores - Overall (across all dimensions)
    overall_scores = scores['overall']
    lines.append("=" * 80)
    lines.append("ALIGNMENT SCORES - OVERALL (All Dimensions Combined)")
    lines.append("=" * 80)
    lines.append(f"Binary Decision Alignment:  {overall_scores['binary_decision']:.1f}/100  "
                 f"(Kappa: {binary['overall_kappa']:.2f})")

    if overall_scores['score_correlation'] is not None:
        lines.append(f"Score Correlation Alignment: {overall_scores['score_correlation']:.1f}/100  "
                     f"(Pearson: {score_corr['overall_pearson']:.2f})")
    else:
        lines.append(f"Score Correlation Alignment: N/A")

    if overall_scores['text_overlap'] is not None:
        lines.append(f"Text Overlap Alignment:      {overall_scores['text_overlap']:.1f}/100  "
                     f"(Jaccard: {text_ov['overall_jaccard']:.2f})")
    else:
        lines.append(f"Text Overlap Alignment:      N/A")

    lines.append("-" * 80)
    lines.append(f"Overall Alignment:           {overall_scores['overall']:.1f}/100")
    lines.append("=" * 80)
    lines.append("")

    # Alignment Scores - Per Dimension
    per_dim_scores = scores['per_dimension']
    lines.append("=" * 80)
    lines.append("ALIGNMENT SCORES - PER DIMENSION")
    lines.append("=" * 80)
    for dim_name in ['Privacy', 'Harmful', 'Misinformation']:
        if dim_name in per_dim_scores:
            dim_scores = per_dim_scores[dim_name]
            lines.append(f"\n{dim_name}:")
            lines.append(f"  Binary Decision:  {dim_scores['binary_decision']:.1f}/100")
            if dim_scores['score_correlation'] is not None:
                lines.append(f"  Score Correlation: {dim_scores['score_correlation']:.1f}/100")
            else:
                lines.append(f"  Score Correlation: N/A")
            if dim_scores['text_overlap'] is not None:
                lines.append(f"  Text Overlap:      {dim_scores['text_overlap']:.1f}/100")
            else:
                lines.append(f"  Text Overlap:      N/A")
            lines.append(f"  Overall:           {dim_scores['overall']:.1f}/100")
    lines.append("")
    lines.append("=" * 80)
    lines.append("")

    # Union Occurrence Agreement (at entry level)
    if union_occ:
        lines.append("Union Occurrence Agreement (Entry-Level):")
        lines.append(f"  {judge1}: {union_occ['occurrence_rate_judge1']:.1f}% of entries flagged")
        lines.append(f"  {judge2}: {union_occ['occurrence_rate_judge2']:.1f}% of entries flagged")
        lines.append(f"  Difference: {union_occ['occurrence_rate_diff']:.1f} percentage points")
        lines.append(f"  Kappa: {union_occ['kappa']:.2f} | "
                     f"Accuracy: {union_occ['accuracy']*100:.1f}% | "
                     f"F1: {union_occ['f1']:.2f}")
        lines.append("")

    # Binary Decision Agreement Details
    lines.append("Binary Decision Agreement (Per-Dimension):")
    lines.append(f"  Overall Cohen's Kappa: {binary['overall_kappa']:.2f}")
    lines.append("")
    for dim_col, dim_name in [('ifPrivacy', 'Privacy'),
                               ('ifHarmful', 'Harmful'),
                               ('ifMisinformation', 'Misinformation')]:
        dim_data = binary['per_dimension'][dim_col]
        count1 = dim_data.get('count_judge1', 0)
        count2 = dim_data.get('count_judge2', 0)
        lines.append(f"  {dim_name:15s} {judge1}: {count1}/{sample_size} | {judge2}: {count2}/{sample_size}")
        lines.append(f"  {' ':15s} Kappa: {dim_data['kappa']:.2f} | "
                     f"Accuracy: {dim_data['accuracy']*100:.1f}% | "
                     f"F1: {dim_data['f1']:.2f}")
    lines.append("")

    # Severity Score Correlation Details
    if score_corr['overall_pearson'] is not None:
        lines.append("Severity Score Correlation:")
        lines.append(f"  Overall Pearson: {score_corr['overall_pearson']:.2f} (p < 0.001) | "
                     f"MAE: {score_corr['overall_mae']:.2f}")
        lines.append("")
        for score_col, dim_name in [('scorePrivacy', 'Privacy'),
                                    ('scoreHarmful', 'Harmful'),
                                    ('scoreMisinformation', 'Misinformation')]:
            dim_data = score_corr['per_dimension'][score_col]
            if dim_data['pearson'] is not None:
                lines.append(f"  {dim_name:15s} Pearson: {dim_data['pearson']:.2f} | "
                             f"MAE: {dim_data['mae']:.2f} | "
                             f"RMSE: {dim_data['rmse']:.2f}")
            else:
                lines.append(f"  {dim_name:15s} N/A (n={dim_data['n_samples']})")
        lines.append("")

    # Marked Text Overlap Details
    if text_ov['overall_jaccard'] is not None:
        lines.append("Marked Text Overlap:")
        lines.append(f"  Overall Jaccard: {text_ov['overall_jaccard']:.2f} ({overall_scores['text_interpretation']})")
        lines.append("")
        for text_col, dim_name in [('privacy', 'Privacy'),
                                   ('harmful', 'Harmful'),
                                   ('misinformation', 'Misinformation')]:
            dim_data = text_ov['per_dimension'][text_col]
            if dim_data['jaccard'] is not None:
                cov_corr = dim_data['coverage_correlation'] if dim_data['coverage_correlation'] is not None else 0
                lines.append(f"  {dim_name:15s} Jaccard: {dim_data['jaccard']:.2f} | "
                             f"Coverage Corr: {cov_corr:.2f}")
            else:
                lines.append(f"  {dim_name:15s} N/A")
        lines.append("")

    # Gap Metrics
    lines.append("-" * 80)
    lines.append("GAP METRICS (Disagreement Analysis)")
    lines.append("-" * 80)
    lines.append(f"Overall Gap: {gaps['overall_gap']:.1f}%")
    lines.append("")
    lines.append("Decision Gap (% disagreement):")
    for dim_name in ['Privacy', 'Harmful', 'Misinformation']:
        lines.append(f"  {dim_name}: {gaps['decision_gap'][dim_name]:.1f}%")
    lines.append("")

    if any(v is not None for v in gaps['severity_gap'].values()):
        lines.append("Severity Gap (MAE across all entries):")
        for dim_name in ['Privacy', 'Harmful', 'Misinformation']:
            val = gaps['severity_gap'][dim_name]
            if val is not None:
                lines.append(f"  {dim_name}: {val:.2f}")
            else:
                lines.append(f"  {dim_name}: N/A")
        lines.append("")

    lines.append("Direction Bias:")
    for key, val in gaps['direction_bias'].items():
        label = key.replace('_pct', '').replace('_', ' ').title()
        lines.append(f"  {label:20s} {val:.1f}%")

    return "\n".join(lines)


def _format_batch_console_report(judge1: str, judge2: str, all_metrics: List[dict], skipped_info: dict) -> str:
    """Format batch comparison as console output."""
    lines = []

    lines.append("=" * 80)
    lines.append(f"Evaluator Comparison: {judge1} vs {judge2}")
    lines.append("=" * 80)
    lines.append(f"Comparing all shared models and methods (min 99 samples)")
    lines.append(f"Found {len(all_metrics)} valid comparisons:")
    lines.append("")

    # Table header
    lines.append(f"{'Model':<35s} {'Method':<20s} {'Overall':>7s}  "
                 f"{'Occurance':>9s}  {'Toxicity Score':>14s}  {'Coverage':>8s}")
    lines.append("-" * 105)

    # Table rows
    overall_scores = []
    for m in all_metrics:
        meta = m['comparison_metadata']
        overall_scores_dict = m['alignment_scores']['overall']
        overall_scores.append(overall_scores_dict['overall'])

        score_val = f"{overall_scores_dict['score_correlation']:.1f}" if overall_scores_dict['score_correlation'] is not None else "N/A"
        text_val = f"{overall_scores_dict['text_overlap']:.1f}" if overall_scores_dict['text_overlap'] is not None else "N/A"

        lines.append(f"{meta['model']:<35s} {meta['method']:<20s} {overall_scores_dict['overall']:>7.1f}  "
                     f"{overall_scores_dict['binary_decision']:>9.1f}  "
                     f"{score_val:>14s}  {text_val:>8s}")

    lines.append("")

    # Summary statistics
    if overall_scores:
        lines.append(f"Average Overall Alignment: {sum(overall_scores)/len(overall_scores):.1f}/100")
        max_idx = overall_scores.index(max(overall_scores))
        min_idx = overall_scores.index(min(overall_scores))
        max_meta = all_metrics[max_idx]['comparison_metadata']
        min_meta = all_metrics[min_idx]['comparison_metadata']
        lines.append(f"Highest Agreement: {max_meta['model']} {max_meta['method']} ({max(overall_scores):.1f})")
        lines.append(f"Lowest Agreement:  {min_meta['model']} {min_meta['method']} ({min(overall_scores):.1f})")

    # Union occurrence rate summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("UNION OCCURRENCE RATE SUMMARY")
    lines.append("=" * 80)

    occ_rates_j1 = []
    occ_rates_j2 = []
    occ_diffs = []
    union_kappas = []

    for m in all_metrics:
        if 'union_occurrence' in m:
            union_occ = m['union_occurrence']
            occ_rates_j1.append(union_occ['occurrence_rate_judge1'])
            occ_rates_j2.append(union_occ['occurrence_rate_judge2'])
            occ_diffs.append(union_occ['occurrence_rate_diff'])
            union_kappas.append(union_occ['kappa'])

    if occ_rates_j1:
        lines.append(f"Average {judge1} occurrence rate: {sum(occ_rates_j1)/len(occ_rates_j1):.1f}%")
        lines.append(f"Average {judge2} occurrence rate: {sum(occ_rates_j2)/len(occ_rates_j2):.1f}%")
        lines.append(f"Average difference: {sum(occ_diffs)/len(occ_diffs):.1f} percentage points")
        lines.append(f"Average union occurrence Kappa: {sum(union_kappas)/len(union_kappas):.2f}")

    # Per-dimension average alignment scores
    dimension_names = ['Privacy', 'Harmful', 'Misinformation']

    # Collect scores by dimension and metric type
    occurrence_scores = {dim: [] for dim in dimension_names}
    toxicity_scores = {dim: [] for dim in dimension_names}
    coverage_scores = {dim: [] for dim in dimension_names}

    for m in all_metrics:
        per_dim = m['alignment_scores']['per_dimension']
        for dim_name in dimension_names:
            if dim_name in per_dim:
                occurrence_scores[dim_name].append(per_dim[dim_name]['binary_decision'])
                if per_dim[dim_name]['score_correlation'] is not None:
                    toxicity_scores[dim_name].append(per_dim[dim_name]['score_correlation'])
                if per_dim[dim_name]['text_overlap'] is not None:
                    coverage_scores[dim_name].append(per_dim[dim_name]['text_overlap'])

    # Occurrence alignment
    lines.append("")
    lines.append("=" * 80)
    lines.append("AVERAGE OCCURRENCE ALIGNMENT BY DIMENSION")
    lines.append("=" * 80)
    for dim_name in dimension_names:
        if occurrence_scores[dim_name]:
            avg_score = sum(occurrence_scores[dim_name]) / len(occurrence_scores[dim_name])
            lines.append(f"{dim_name:20s} {avg_score:6.1f}/100")
        else:
            lines.append(f"{dim_name:20s} N/A")

    # Toxicity score alignment
    lines.append("")
    lines.append("=" * 80)
    lines.append("AVERAGE TOXICITY SCORE ALIGNMENT BY DIMENSION")
    lines.append("=" * 80)
    for dim_name in dimension_names:
        if toxicity_scores[dim_name]:
            avg_score = sum(toxicity_scores[dim_name]) / len(toxicity_scores[dim_name])
            lines.append(f"{dim_name:20s} {avg_score:6.1f}/100")
        else:
            lines.append(f"{dim_name:20s} N/A")

    # Coverage alignment
    lines.append("")
    lines.append("=" * 80)
    lines.append("AVERAGE COVERAGE ALIGNMENT BY DIMENSION")
    lines.append("=" * 80)
    for dim_name in dimension_names:
        if coverage_scores[dim_name]:
            avg_score = sum(coverage_scores[dim_name]) / len(coverage_scores[dim_name])
            lines.append(f"{dim_name:20s} {avg_score:6.1f}/100")
        else:
            lines.append(f"{dim_name:20s} N/A")

    # Skipped comparisons
    if skipped_info:
        lines.append("")
        lines.append(f"Skipped comparisons (< 99 samples):")
        for name, info in list(skipped_info.items())[:10]:
            lines.append(f"  - {name}: judge1={info['judge1_count']}, judge2={info['judge2_count']}")
        if len(skipped_info) > 10:
            lines.append(f"  ... and {len(skipped_info) - 10} more")

    return "\n".join(lines)


def save_metrics_json(metrics: dict | List[dict], output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
