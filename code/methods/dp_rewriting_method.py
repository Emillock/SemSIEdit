import json
import os
import sys

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.normpath(os.path.join(current_dir, '..'))
tools_path = os.path.normpath(os.path.join(code_dir, 'tools'))
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)
if tools_path not in sys.path:
    sys.path.insert(0, tools_path)

from tqdm import tqdm
from utils import run_with_retry, run_without_retry
from dp_rewriting import DPRewritingTool


def generate_dp_rewriting(model_name: str, source_name: str, max_lines: int = -1) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    if model_name.startswith("hf:"):
        model_display_name = model_name.split('/')[-1]
    else:
        model_display_name = model_name.split('/')[1].removesuffix(":free") if "/" in model_name else model_name

    # Derive source model name from the source file for the output name
    source_basename = os.path.basename(source_name)
    source_model = source_basename.replace("-Baseline.jsonl", "")

    out_name = "datasets/" + source_model + "-DP_Rewriting.jsonl"
    log_name = out_name.replace("datasets/", "logs/")

    done_ids = []

    if not os.path.exists(os.path.dirname(log_name)):
        os.makedirs(os.path.dirname(log_name), exist_ok=True)

    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            for line in f:
                done_ids.append(json.loads(line)["ID"])

    dp_tool = DPRewritingTool()

    def generate(token: str) -> None:
        for data in tqdm(dataset):
            try:
                if data["ID"] in done_ids:
                    print(f'Skip {data["ID"]}')
                    continue

                answer = data.get("answer", "")
                if not answer:
                    print(f"Entry {data['ID']} has no answer, skipping")
                    continue

                print(f"\nProcessing entry {data['ID']}")

                result_json = dp_tool._run(
                    text=answer,
                    model_name=model_name,
                    api_token=token or "",
                )
                result = json.loads(result_json)

                data['old_answer'] = answer
                data['answer'] = result.get('rewritten_text', answer)
                data['dp_privacy_metrics'] = result.get('privacy_metrics', {})

                debug_data = {
                    "original_answer": answer,
                    "dp_answer": data['answer'],
                    "applied_strategies": result.get('applied_strategies', []),
                    "privacy_metrics": data['dp_privacy_metrics'],
                    "error": result.get('error', None),
                }
            except Exception as e:
                msg = f"DP rewriting error: {str(e)}"
                debug_data = {"error": [msg]}
                data['old_answer'] = answer
                data['dp_privacy_metrics'] = {}
                print(msg)

            with open(out_name, 'a') as file2:
                json_line = json.dumps(data)
                file2.write(json_line + '\n')

            with open(log_name, 'a', encoding='utf-8') as f:
                f.write(json.dumps(debug_data, ensure_ascii=False) + '\n')

            done_ids.append(data["ID"])
        print("All done!")

    if model_name.startswith("hf:"):
        run_without_retry(generate)
    else:
        run_with_retry(generate)


if __name__ == '__main__':
    model_name = "qwen/qwen3-8b"
    source = "./datasets/gpt-5-Baseline.jsonl"
    max_lines = -1

    generate_dp_rewriting(model_name, source, max_lines)
