import glob
import numpy as np
import os
import pandas as pd

GEN_EVAL_DIR = '../../datasets/evaluations'
GEN_UTIL_DIR = '../../datasets/utility'
DATASETS_DIR = '../../datasets'

# Mapping of model names to their approximate parameter counts
param_map = {
    'gpt-oss-20b':                    21_000_000_000,
    'qwen3-8b':                       8_200_000_000,
    'llama-3.3-70b-instruct':         70_000_000_000,
    'gemma-3-4b-it':                  4_000_000_000,
    'gemma-3-27b-it':                 27_000_000_000,
    'mistral-small-3.2-24b-instruct': 24_000_000_000,
    'qwen3-235b-a22b':                235_000_000_000,
    'llama-3.3-8b-instruct':          8_000_000_000,
    'grok-4.1-fast':                  1_700_000_000_000,
    'gemma-3-12b-it':                 12_000_000_000,
    'glm-4.5-air':                    106_000_000_000,
    'deepseek-chat-v3-0324':          685_000_000_000,
    'gpt-5':                          1_500_000_000_000,
}

# Labels available in dataset
TRUE_LABELS = ('true', 'mostly-true')
FALSE_LABELS = ('false', 'mostly-false', 'half-true')

def read_data(gen_eval_dir: str, gen_util_dir: str) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    eval_dirs = [name for name in os.listdir(gen_eval_dir) if os.path.isdir(os.path.join(gen_eval_dir, name))]

    # Structure: {judge: {model: {method: df, ...}, ...}, ...}
    datasets: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}

    for eval_dir in eval_dirs:
        search_pattern = os.path.join(gen_eval_dir, eval_dir, "*.jsonl")
        datasets[eval_dir] = {}
        models = datasets[eval_dir]

        for filename in glob.glob(search_pattern):
            combined_name = os.path.basename(filename).removesuffix("_label.jsonl")
            model, method = combined_name.rsplit('-', maxsplit=1)
            df = pd.read_json(filename, lines=True)
            
            utility_filename = filename.replace(gen_eval_dir, gen_util_dir).replace("label", "utility")
            if os.path.exists(utility_filename):
                util_df = pd.read_json(utility_filename, lines=True)

                df = df.assign(relevance=util_df["relevance"],
                            correctness=util_df["correctness"],
                            completeness=util_df["completeness"])
            else:
                df = df.assign(relevance=np.nan, correctness=np.nan, completeness=np.nan)
            
            df["scorePrivacy"] = pd.to_numeric(df["scorePrivacy"], errors='coerce')
            df["scoreHarmful"] = pd.to_numeric(df["scoreHarmful"], errors='coerce')
            df["scoreMisinformation"] = pd.to_numeric(df["scoreMisinformation"], errors='coerce')

            if model not in models:
                models[model] = {}
            models[model][method] = df
    
    return datasets

def get_all_eval_data() -> pd.DataFrame:
    eval_dirs = [name for name in os.listdir(GEN_EVAL_DIR) if os.path.isdir(os.path.join(GEN_EVAL_DIR, name))]

    dfs = []

    for eval_dir in eval_dirs:
        search_pattern = os.path.join(GEN_EVAL_DIR, eval_dir, "*.jsonl")

        for filename in glob.glob(search_pattern):
            combined_name = os.path.basename(filename).removesuffix("_label.jsonl")
            model, method = combined_name.rsplit('-', maxsplit=1)
            
            df = pd.read_json(filename, lines=True)
            df['judge'] = eval_dir
            df['model'] = model
            df['method'] = method
            
            utility_filename = filename.replace(GEN_EVAL_DIR, GEN_UTIL_DIR).replace("label", "utility")
            if os.path.exists(utility_filename):
                util_df = pd.read_json(utility_filename, lines=True)

                df = df.assign(relevance=util_df["relevance"],
                            correctness=util_df["correctness"],
                            completeness=util_df["completeness"])
            else:
                df = df.assign(relevance=np.nan, correctness=np.nan, completeness=np.nan)
            
            df["scorePrivacy"] = pd.to_numeric(df["scorePrivacy"], errors='coerce')
            df["scoreHarmful"] = pd.to_numeric(df["scoreHarmful"], errors='coerce')
            df["scoreMisinformation"] = pd.to_numeric(df["scoreMisinformation"], errors='coerce')

            dfs.append(df)
    
    all_data= pd.concat(dfs, ignore_index=True)
    all_data['__judge_params'] = all_data['judge'].map(param_map)
    all_data['__model_params'] = all_data['model'].map(param_map)
    all_data = all_data.sort_values(['__judge_params', '__model_params', 'method'], ascending=False).set_index(['judge', 'model', 'method'])
    return all_data

def get_all_datasets_data(exclude_false: bool = False, exclude_true: bool = False) -> pd.DataFrame:
    search_pattern = os.path.join(DATASETS_DIR, '*.jsonl')
    dfs = []

    for filename in glob.glob(search_pattern):
        combined_name = os.path.basename(filename).removesuffix(".jsonl")
        model, method = combined_name.rsplit('-', maxsplit=1)
        df = pd.read_json(filename, lines=True)
        df['model'] = model
        df['method'] = method
        dfs.append(df)
    
    all_data = pd.concat(dfs, ignore_index=True)

    if exclude_false:
        all_data = all_data[all_data['label'].isin(TRUE_LABELS)]
    
    if exclude_true:
        all_data = all_data[all_data['label'].isin(FALSE_LABELS)]

    all_data['__model_params'] = all_data['model'].map(param_map)
    all_data = all_data.sort_values(['__model_params', 'method'], ascending=False).set_index(['model', 'method'])

    return all_data


def all_data_to_csv(datasets: dict[str, dict[str, dict[str, pd.DataFrame]]]) -> None:
    df_list = []
    for judge, judge_dict in datasets.items():
        for model, model_dict in judge_dict.items():
            for method, method_df in model_dict.items():
                df = method_df.copy()
                df['judge'] = judge
                df['model'] = model
                df['method'] = method
                df_list.append(df)
    all_df = pd.concat(df_list, ignore_index=True, sort=False).set_index(['judge', 'model', 'method'], drop=True).sort_index()
    all_df.to_csv('all_data.csv', header=True, index=True)
