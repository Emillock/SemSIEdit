import json
import os

from utils import *
from tqdm import tqdm

def generate_baseline(model_name: str, source_name: str, max_lines: int = -1) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    out_name = "datasets/" + model_name.split('/')[1].removesuffix(":free") + "-Baseline.jsonl"
    log_name = out_name.replace("datasets/", "logs/")

    done_ids = []
    
    if not os.path.exists(log_name):
        os.makedirs(os.path.dirname(log_name), exist_ok=True)

    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            for line in f:
                done_ids.append(json.loads(line)["ID"])

    def generate(token: str) -> None:
        for data in tqdm(dataset):
            try:
                if data["ID"] in done_ids:
                    print(f'Skip {data["ID"]}')
                    continue

                statement = data['statement'][:-1] if data['statement'][-1] == '.' else data['statement']
                prompt = f"Why {statement}?"
                
                print("\nsend request")
                print("prompt:", prompt)

                answer, reasoning = generate_answer(model_name, token, prompt)
                print(answer)

                data['prompt'] = prompt
                data['model'] = model_name
                data['answer'] = answer

                debug_data = {"reasoning": reasoning, "answer": answer}
            except openai.PermissionDeniedError as e:
                msg = f"Permission denied: {e.body['message']}"
                debug_data = {"error": [msg]}
                data["error"] = [msg]
                print(msg)
            
            with open(out_name, 'a') as file2:
                json_line = json.dumps(data)
                file2.write(json_line + '\n')

            with open(log_name, 'a', encoding='utf-8') as f:
                f.write(json.dumps(debug_data, ensure_ascii=False) + '\n')
        
            done_ids.append(data["ID"])
        print("All done!")
    
    run_with_retry(generate)


if __name__ == '__main__':
    model_name = "qwen/qwen3-235b-a22b:free"
    max_lines = 100
    
    generate_baseline(model_name, "./semsi-datasets/gemini1.0-pro_label.jsonl", max_lines)
