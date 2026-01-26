import os
import sys
from utils import run_evaluation_with_retry, run_with_retry
from metric_merge import evaluate as evaluate_semsi_openai
from metric_merge_ollama import evaluate as evaluate_semsi_ollama
from utility_evaluation import evaluate_utility

tokens = os.environ.get("TOKENS", "").split(",")
seed = os.environ.get("SHUFFLE_SEED")

if __name__ == "__main__":

    eval_model = sys.argv[1]
    # Set up backlog of datasets to evaluate
    backlog = [
                # "example-dataset.jsonl",
    ]

    evaluate = evaluate_semsi_openai if '/' in eval_model else evaluate_semsi_ollama

    for datafilename in backlog:
        datafilename = f"./datasets/{datafilename}"
        if os.path.exists(datafilename):
            if not run_evaluation_with_retry(evaluate, datafilename, eval_model, tokens=tokens):
                break
            print(f"Finish eval: {datafilename}")

    for datafilename in backlog:
        datafilename = f"./datasets/{datafilename}"
        if os.path.exists(datafilename):
            run_with_retry(evaluate_utility, datafilename, eval_model, tokens=tokens)
            print(f"Finish eval: {datafilename}")
