"""
Example script demonstrating evaluator similarity comparison.

This script shows various ways to use the evaluator_analysis module.
"""

from evaluator_analysis import compare_evaluators
import pandas as pd


def single_comparison():
    """Compare specific model/method between two judges."""

    # Compare gpt-5 vs gpt-oss-20b on gpt-5 Baseline
    result = compare_evaluators(
        judge1='gpt-5',
        judge2='gpt-oss-20b',
        model='gpt-5',
        method='Baseline',
        output_format='console'
    )

    print("\nExample 1 complete!\n")


def batch_comparison():
    """Compare all shared models/methods."""

    # Get results as dict for processing
    results = compare_evaluators(
        judge1='gpt-5',
        judge2='gpt-oss-20b',
        output_format='dict',
        min_samples=99
    )

    # Extract key metrics
    print(f"\nFound {len(results)} valid comparisons\n")
    print(f"{'Model':<25s} {'Method':<20s} {'Alignment':>10s}")
    print("-" * 60)

    for r in results[:10]:  # Show first 10
        meta = r['comparison_metadata']
        score = r['alignment_scores']['overall']['overall']
        print(f"{meta['model']:<25s} {meta['method']:<20s} {score:>10.1f}")

    if len(results) > 10:
        print(f"... and {len(results) - 10} more")

    print("\nComplete!\n")


def main():
    """Run all examples."""

    # single_comparison()
    batch_comparison()



if __name__ == '__main__':
    main()
