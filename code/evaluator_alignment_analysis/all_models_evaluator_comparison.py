"""
Compare internal evaluators of all models against gpt-5 judge.

Prints a table with occurrence correlation, toxicity score correlation,
coverage correlation, and SemSI occurrence rates (baseline vs after editing).
"""

import sys
import io
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from evaluator_analysis import compare_evaluators


def _find_repo_root():
    repo_root = Path(__file__).parent.resolve()
    while repo_root != repo_root.parent:
        if (repo_root / 'datasets').exists():
            break
        repo_root = repo_root.parent
    return repo_root


def get_semsi_edit_occurrence(model, judge='gpt-5', method='Feedback(3Iter)'):
    """Compute SemSI occurrence rate from judge evaluation of edited output."""
    fpath = _find_repo_root() / 'datasets' / 'evaluations' / judge / f'{model}-{method}_label.jsonl'
    with open(fpath, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    valid = [e for e in entries if e.get('error') is None]
    flagged = sum(1 for e in valid
                  if e.get('ifPrivacy') == 'yes'
                  or e.get('ifHarmful') == 'yes'
                  or e.get('ifMisinformation') == 'yes')
    return flagged / len(valid) * 100 if valid else 0.0


def get_dimension_edit_occurrence(model, field, judge='gpt-5', method='Feedback(3Iter)'):
    """Compute per-dimension occurrence rate from judge evaluation of edited output."""
    fpath = _find_repo_root() / 'datasets' / 'evaluations' / judge / f'{model}-{method}_label.jsonl'
    with open(fpath, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    valid = [e for e in entries if e.get('error') is None]
    flagged = sum(1 for e in valid if e.get(field) == 'yes')
    return flagged / len(valid) * 100 if valid else 0.0


# (model, method) pairs — gpt-5 appears twice with different methods
MODEL_METHODS = [
    ("deepseek-chat-v3-0324",            "Feedback(3Iter)"),
    ("gemma-3-12b-it",                   "Feedback(3Iter)"),
    ("gemma-3-27b-it",                   "Feedback(3Iter)"),
    ("gemma-3-4b-it",                    "Feedback(3Iter)"),
    ("glm-4.5-air",                      "Feedback(3Iter)"),
    ("gpt-5",                            "Feedback(3Iter)"),
    ("gpt-5",                            "FeedbackFullEval(3Iter)"),
    ("gpt-oss-20b",                      "Feedback(3Iter)"),
    ("grok-4.1-fast",                    "Feedback(3Iter)"),
    ("llama-3.3-70b-instruct",           "Feedback(3Iter)"),
    ("llama-3.3-8b-instruct",            "Feedback(3Iter)"),
    ("mistral-small-3.2-24b-instruct",   "Feedback(3Iter)"),
    ("qwen3-235b-a22b",                  "Feedback(3Iter)"),
    ("qwen3-8b",                         "Feedback(3Iter)"),
]

HDR = f"{'Model':<35s} {'Method':<25s} {'Occur.corr.':>11s} {'Occur.agr.':>10s} {'Tox.Score corr.':>16s} {'Cov.Corr.':>10s} {'Int.Eval.Occ.':>14s} {'Baseline Occ.':>14s} {'SemSIEdit Occ.':>15s} {'Diff%':>7s}"
SEP = "-" * 163


def fmt_row(model, method, kappa, accuracy, pearson, jaccard, occ_int, occ_jdg, edit_occ, diff):
    return (
        f"{model:<35s} {method:<25s} {kappa:>11.2f} {accuracy:>9.1%} "
        f"{pearson:>16.2f} {jaccard:>10.2f} {occ_int:>13.1f}% "
        f"{occ_jdg:>13.1f}% {edit_occ:>14.1f}% {diff:>+6.1f}%"
    )


def print_kappa_diff_correlation(kappas, diffs, label=""):
    valid = [(k, d) for k, d in zip(kappas, diffs) if not (np.isnan(k) or np.isnan(d))]
    if len(valid) < 2:
        print("  (not enough valid data points for correlation)")
        return
    ks, ds = zip(*valid)
    pearson_r,  pearson_p  = pearsonr(ks, ds)
    spearman_r, spearman_p = spearmanr(ks, ds)
    tag = f" - {label}" if label else ""
    print(f"\n  Correlation (Occur.corr. vs Diff%){tag}")
    print(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.4f})")
    print(f"  Spearman r = {spearman_r:+.4f}  (p = {spearman_p:.4f})")
    print(f"  n = {len(valid)} models")


def main():
    results = []  # (model, method, r, edit_occ)
    for model, method in MODEL_METHODS:
        try:
            r = compare_evaluators(
                judge1='gpt-5',
                model=model,
                method=method,
                mode='evaluator',
                output_format='dict'
            )
            edit_occ = get_semsi_edit_occurrence(model, method=method)
            results.append((model, method, r, edit_occ))
        except Exception as e:
            import traceback
            print(f"SKIP {model} [{method}]: {e}")
            traceback.print_exc()

    # ── Overall table ──────────────────────────────────────────────────────────
    print()
    print(HDR)
    print(SEP)

    overall_kappas, overall_diffs = [], []
    for model, method, r, edit_occ in sorted(
        results,
        key=lambda x: x[2]['binary_agreement']['overall_kappa'],
        reverse=True,
    ):
        binary = r['binary_agreement']
        score  = r['score_correlation']
        text   = r['text_overlap']
        union  = r.get('union_occurrence', {})

        kappa    = binary['overall_kappa']
        dims     = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
        accuracy = sum(binary['per_dimension'].get(d, {}).get('accuracy', 0) for d in dims) / len(dims)
        pearson  = score['overall_pearson'] if score['overall_pearson'] is not None else float('nan')
        jaccard  = text['overall_jaccard']  if text['overall_jaccard']  is not None else float('nan')
        occ_int  = union.get('occurrence_rate_judge1', float('nan'))
        occ_jdg  = union.get('occurrence_rate_judge2', float('nan'))
        diff     = ((edit_occ - occ_jdg) / occ_jdg * 100) if occ_jdg > 0 else 0.0

        overall_kappas.append(kappa)
        overall_diffs.append(diff)

        print(fmt_row(model, method, kappa, accuracy, pearson, jaccard, occ_int, occ_jdg, edit_occ, diff))

    print_kappa_diff_correlation(overall_kappas, overall_diffs, label="Overall")

    # ── Per-dimension tables ───────────────────────────────────────────────────
    dimensions = [
        ('Privacy',         'ifPrivacy',         'scorePrivacy',         'privacy'),
        ('Harmful',         'ifHarmful',          'scoreHarmful',         'harmful'),
        ('Misinformation',  'ifMisinformation',   'scoreMisinformation',  'misinformation'),
    ]

    for dim_name, binary_key, score_key, text_key in dimensions:
        print()
        print(f"=== {dim_name.upper()} ===")
        print()
        print(HDR)
        print(SEP)

        dim_kappas, dim_diffs = [], []
        for model, method, r, _ in sorted(
            results,
            key=lambda x: x[2]['binary_agreement']['per_dimension'].get(binary_key, {}).get('kappa', -2),
            reverse=True,
        ):
            b = r['binary_agreement']['per_dimension'].get(binary_key, {})
            s = r['score_correlation']['per_dimension'].get(score_key, {})
            t = r['text_overlap']['per_dimension'].get(text_key, {})

            kappa    = b.get('kappa', float('nan'))
            accuracy = b.get('accuracy', 0)
            pearson  = s.get('pearson') if s.get('pearson') is not None else float('nan')
            jaccard  = t.get('jaccard') if t.get('jaccard') is not None else float('nan')
            occ_int  = b.get('count_judge1', 0)
            occ_jdg  = b.get('count_judge2', 0)
            n        = r['comparison_metadata']['sample_size']
            occ_int_pct = occ_int / n * 100 if n else 0.0
            occ_jdg_pct = occ_jdg / n * 100 if n else 0.0
            edit_occ    = get_dimension_edit_occurrence(model, binary_key, method=method)
            diff        = ((edit_occ - occ_jdg_pct) / occ_jdg_pct * 100) if occ_jdg_pct > 0 else 0.0

            dim_kappas.append(kappa)
            dim_diffs.append(diff)

            print(fmt_row(model, method, kappa, accuracy, pearson, jaccard, occ_int_pct, occ_jdg_pct, edit_occ, diff))

        print_kappa_diff_correlation(dim_kappas, dim_diffs, label=dim_name)


if __name__ == '__main__':
    main()