"""
Data loading and alignment utilities for evaluator comparison.
"""

import ast
import os
import glob
import json
import pandas as pd
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the repository root by searching for .git directory."""
    current = Path(__file__).parent.resolve()
    while current != current.parent:
        if (current / '.git').exists() or (current / 'datasets').exists():
            return current
        current = current.parent
    # Fallback: assume we're somewhere under the repo
    return Path(__file__).parent.resolve()


def find_shared_model_methods(
    judge1: str,
    judge2: str,
    eval_dir: str | None = None
) -> list[tuple[str, str]]:
    """
    Find all (model, method) pairs that exist in both judge directories.

    Args:
        judge1: First judge name
        judge2: Second judge name
        eval_dir: Path to evaluations directory (default: auto-detect repo_root/datasets/evaluations)

    Returns:
        List of (model, method) tuples that exist for both judges
    """
    # Auto-detect path if not provided
    if eval_dir is None:
        repo_root = _find_repo_root()
        eval_path = repo_root / 'datasets' / 'evaluations'
    else:
        # Get absolute path relative to this file's location
        script_dir = Path(__file__).parent
        eval_path = (script_dir / eval_dir).resolve()

    judge1_dir = eval_path / judge1
    judge2_dir = eval_path / judge2

    if not judge1_dir.exists():
        raise FileNotFoundError(f"Judge directory not found: {judge1_dir}")
    if not judge2_dir.exists():
        raise FileNotFoundError(f"Judge directory not found: {judge2_dir}")

    # Find all evaluation files for each judge
    def get_model_methods(judge_dir):
        model_methods = set()
        pattern = str(judge_dir / "*_label.jsonl")
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            # Remove _label.jsonl suffix
            combined_name = filename.removesuffix("_label.jsonl")
            # Split into model and method
            if '-' in combined_name:
                parts = combined_name.rsplit('-', maxsplit=1)
                if len(parts) == 2:
                    model, method = parts
                    model_methods.add((model, method))
        return model_methods

    judge1_pairs = get_model_methods(judge1_dir)
    judge2_pairs = get_model_methods(judge2_dir)

    # Find intersection
    shared_pairs = sorted(judge1_pairs & judge2_pairs)

    return shared_pairs


def filter_by_min_samples(
    judge1: str,
    judge2: str,
    model_method_pairs: list[tuple[str, str]],
    min_samples: int = 99,
    eval_dir: str | None = None
) -> tuple[list[tuple[str, str]], dict]:
    """
    Filter model/method pairs by minimum sample count.

    Args:
        judge1: First judge name
        judge2: Second judge name
        model_method_pairs: List of (model, method) tuples to check
        min_samples: Minimum number of valid samples required
        eval_dir: Path to evaluations directory (default: auto-detect)

    Returns:
        Tuple of (filtered_pairs, skipped_info) where skipped_info contains
        details about skipped pairs
    """
    # Auto-detect path if not provided
    if eval_dir is None:
        repo_root = _find_repo_root()
        eval_path = repo_root / 'datasets' / 'evaluations'
    else:
        script_dir = Path(__file__).parent
        eval_path = (script_dir / eval_dir).resolve()

    valid_pairs = []
    skipped_info = {}

    for model, method in model_method_pairs:
        # Construct file paths
        file1 = eval_path / judge1 / f"{model}-{method}_label.jsonl"
        file2 = eval_path / judge2 / f"{model}-{method}_label.jsonl"

        try:
            # Load and count valid entries
            df1 = pd.read_json(file1, lines=True)
            df2 = pd.read_json(file2, lines=True)

            # Filter out error records
            def count_valid(df):
                if 'error' in df.columns:
                    valid = df[(df['error'].isna()) | (df['error'] == None) | (df['error'].apply(lambda x: isinstance(x, float) and pd.isna(x)))]
                    return len(valid)
                return len(df)

            count1 = count_valid(df1)
            count2 = count_valid(df2)

            # Check if both meet minimum
            if count1 >= min_samples and count2 >= min_samples:
                valid_pairs.append((model, method))
            else:
                skipped_info[f"{model}-{method}"] = {
                    'judge1_count': count1,
                    'judge2_count': count2,
                    'reason': f"Insufficient samples (need {min_samples})"
                }

        except Exception as e:
            skipped_info[f"{model}-{method}"] = {
                'judge1_count': None,
                'judge2_count': None,
                'reason': f"Error loading files: {str(e)}"
            }

    return valid_pairs, skipped_info


def load_evaluator_pair(
    judge1: str,
    judge2: str,
    model: str,
    method: str,
    eval_dir: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Load and align evaluation pair from two judges.

    Args:
        judge1: First judge name
        judge2: Second judge name
        model: Model name
        method: Method name
        eval_dir: Path to evaluations directory (default: auto-detect)

    Returns:
        Tuple of (df1, df2, sample_size) where df1 and df2 are aligned
        DataFrames with matching IDs, and sample_size is the number of
        samples used
    """
    # Auto-detect path if not provided
    if eval_dir is None:
        repo_root = _find_repo_root()
        eval_path = repo_root / 'datasets' / 'evaluations'
    else:
        # Get absolute path
        script_dir = Path(__file__).parent
        eval_path = (script_dir / eval_dir).resolve()

    # Construct file paths
    file1 = eval_path / judge1 / f"{model}-{method}_label.jsonl"
    file2 = eval_path / judge2 / f"{model}-{method}_label.jsonl"

    if not file1.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file1}")
    if not file2.exists():
        raise FileNotFoundError(f"Evaluation file not found: {file2}")

    # Load data
    df1 = pd.read_json(file1, lines=True)
    df2 = pd.read_json(file2, lines=True)

    # Filter out error records
    def filter_errors(df):
        if 'error' in df.columns:
            # Keep records where error is null, None, or NaN
            valid = df[
                (df['error'].isna()) |
                (df['error'] == None) |
                (df['error'].apply(lambda x: isinstance(x, float) and pd.isna(x)))
            ].copy()
            return valid
        return df.copy()

    df1 = filter_errors(df1)
    df2 = filter_errors(df2)

    # Ensure required fields exist
    required_fields = ['ID', 'ifPrivacy', 'ifHarmful', 'ifMisinformation']
    for field in required_fields:
        if field not in df1.columns:
            raise ValueError(f"Missing required field '{field}' in {file1}")
        if field not in df2.columns:
            raise ValueError(f"Missing required field '{field}' in {file2}")

    # Convert binary fields to string to ensure consistency
    binary_fields = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
    for field in binary_fields:
        df1[field] = df1[field].astype(str).str.lower()
        df2[field] = df2[field].astype(str).str.lower()

    # Ensure score fields exist with default 0
    score_fields = ['scorePrivacy', 'scoreHarmful', 'scoreMisinformation']
    for field in score_fields:
        if field not in df1.columns:
            df1[field] = 0
        if field not in df2.columns:
            df2[field] = 0
        df1[field] = pd.to_numeric(df1[field], errors='coerce').fillna(0)
        df2[field] = pd.to_numeric(df2[field], errors='coerce').fillna(0)

    # Ensure text fields exist with default ''
    text_fields = ['privacy', 'harmful', 'misinformation']
    for field in text_fields:
        if field not in df1.columns:
            df1[field] = ''
        if field not in df2.columns:
            df2[field] = ''
        df1[field] = df1[field].fillna('')
        df2[field] = df2[field].fillna('')

    # Align by ID (inner join — only keep entries present in both)
    common_ids = set(df1['ID']) & set(df2['ID'])
    df1 = df1[df1['ID'].isin(common_ids)].sort_values('ID').reset_index(drop=True)
    df2 = df2[df2['ID'].isin(common_ids)].sort_values('ID').reset_index(drop=True)
    min_size = len(df1)

    return df1, df2, min_size


def _assemble_from_dimension_answers(
    privacy_str: str,
    harmful_str: str,
    misinfo_str: str,
) -> str:
    """
    Merge three per-dimension evaluator answer strings into one combined dict string.

    FeedbackFullEval logs store evaluations split across three separate fields:
      it{i}_eval_privacy_answer, it{i}_eval_harmful_answer, it{i}_eval_misinformation_answer

    Each is a stringified dict, e.g.:
      "{'ifPrivacy': 'no'}"
      "{'ifHarmful': 'yes', 'scoreHarmful': 2, 'harmful': '<<<...>>>'}"
      "{'ifMisinformation': 'no'}"

    This merges all three into a single dict string that _parse_evaluator_string
    can handle downstream, identical in structure to a standard eval_answer.
    """
    merged: dict = {}
    for raw in (privacy_str, harmful_str, misinfo_str):
        if not raw:
            continue
        try:
            parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                merged.update(parsed)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    merged.update(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
    return str(merged) if merged else ''


def _find_first_evaluator_response(log_entry: dict) -> str | None:
    """Find the first iteration's evaluator response from log entry.

    Supports two log schemas:

    Feedback(3Iter):
        it{i}_eval_answer  — single unified field containing all three dimensions

    FeedbackFullEval(3Iter):
        it{i}_eval_combined          — pre-merged field (preferred)
        it{i}_eval_privacy_answer    — per-dimension fallback (assembled if combined absent)
        it{i}_eval_harmful_answer
        it{i}_eval_misinformation_answer

    In both schemas the first populated iteration (it1, then it2, …) is returned
    so that the internal evaluator's verdict on the *original* answer is used,
    matching the baseline occurrence rate from the judge's evaluation file.
    """
    for i in range(1, 11):
        # Schema 1 — Feedback(3Iter): single unified eval answer
        unified_key = f'it{i}_eval_answer'
        if log_entry.get(unified_key):
            return log_entry[unified_key]

        # Schema 2a — FeedbackFullEval: pre-merged combined field
        combined_key = f'it{i}_eval_combined'
        if log_entry.get(combined_key):
            return log_entry[combined_key]

        # Schema 2b — FeedbackFullEval: assemble from separate dimension answers
        privacy_key = f'it{i}_eval_privacy_answer'
        harmful_key = f'it{i}_eval_harmful_answer'
        misinfo_key = f'it{i}_eval_misinformation_answer'
        if any(log_entry.get(k) for k in (privacy_key, harmful_key, misinfo_key)):
            assembled = _assemble_from_dimension_answers(
                log_entry.get(privacy_key, ''),
                log_entry.get(harmful_key, ''),
                log_entry.get(misinfo_key, ''),
            )
            if assembled:
                return assembled

    return None


def _parse_evaluator_string(eval_str: str) -> dict:
    """Parse internal evaluator string to dict."""
    if not eval_str:
        return {}

    try:
        # If already a dict, return it
        if isinstance(eval_str, dict):
            return eval_str

        # Try ast.literal_eval first (handles Python repr correctly,
        # preserves apostrophes in text like "John's data")
        return ast.literal_eval(eval_str)
    except (ValueError, SyntaxError):
        try:
            # Fallback to JSON parsing
            return json.loads(eval_str)
        except (json.JSONDecodeError, AttributeError, TypeError):
            return {}


def _normalize_text_field(value) -> str:
    """Normalize text fields that may be stored as lists or strings.

    Internal evaluators sometimes return marked text as a list of strings
    (e.g., ['<<<text1>>>', '<<<text2>>>']), while judges return a single
    string ('<<<text1>>> <<<text2>>>'). This normalizes both to a string.
    """
    if isinstance(value, list):
        return ' '.join(str(item) for item in value)
    if value is None:
        return ''
    return str(value)


def load_internal_evaluations(model: str, method: str) -> pd.DataFrame:
    """
    Load internal evaluator responses from feedback loop logs.

    Args:
        model: Model name (e.g., 'llama-3.3-8b-instruct')
        method: Method name (must contain 'Feedback')

    Returns:
        DataFrame with evaluation fields matching judge format
    """
    repo_root = _find_repo_root()
    log_path = repo_root / 'logs' / f'{model}-{method}.jsonl'

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    # Build ID mapping from output file (same order as log, contains original IDs)
    id_mapping = {}  # line_index -> original_ID
    for output_dir in ['datasets', 'datasets-DP']:
        output_path = repo_root / output_dir / f'{model}-{method}.jsonl'
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for out_idx, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        if 'ID' in entry:
                            id_mapping[out_idx] = entry['ID']
                    except json.JSONDecodeError:
                        pass
            break  # Use first found

    records = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            log_entry = json.loads(line)

            # Find first evaluator response — handles both Feedback and FeedbackFullEval schemas
            eval_str = _find_first_evaluator_response(log_entry)
            if eval_str is None:
                continue  # Skip entries without evaluator response

            # Parse evaluator response
            eval_dict = _parse_evaluator_string(eval_str)

            # Resolve real ID: log entry > output file mapping > line index fallback
            real_id = log_entry.get('ID', id_mapping.get(idx, idx))

            # Convert to standard format
            record = {
                'ID': real_id,
                'ifPrivacy': eval_dict.get('ifPrivacy', 'no'),
                'ifHarmful': eval_dict.get('ifHarmful', 'no'),
                'ifMisinformation': eval_dict.get('ifMisinformation', 'no'),
                'scorePrivacy': eval_dict.get('scorePrivacy', 0),
                'scoreHarmful': eval_dict.get('scoreHarmful', 0),
                'scoreMisinformation': eval_dict.get('scoreMisinformation', 0),
                'privacy': _normalize_text_field(eval_dict.get('privacy', '')),
                'harmful': _normalize_text_field(eval_dict.get('harmful', '')),
                'misinformation': _normalize_text_field(eval_dict.get('misinformation', '')),
                'answer': log_entry.get('init_answer', ''),  # Include for reference
                'error': None
            }
            records.append(record)

    df = pd.DataFrame(records)

    # Convert binary fields to string (consistent with load_evaluator_pair)
    binary_fields = ['ifPrivacy', 'ifHarmful', 'ifMisinformation']
    for field in binary_fields:
        df[field] = df[field].astype(str).str.lower()

    # Convert score fields to numeric
    score_fields = ['scorePrivacy', 'scoreHarmful', 'scoreMisinformation']
    for field in score_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    return df