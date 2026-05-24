import json
import os
import requests
import time

from utils import *
from tqdm import tqdm

def evaluate_gpt_oss_safeguard(source_name: str, max_lines: int = -1) -> None:
    with open("./prompts/gpt_oss_safeguard_prompt.txt", 'r') as f:
        dev_instructions = f.read()

    harmony_prompt_raw = (
        f"<|start|>system<|message|>\nReasoning: medium\n<|end|>\n"
        f"<|start|>developer<|message|>\n{dev_instructions}\n"
        "<|start|>user<|message|>\n{prompt}\n<|end|>\n"
        "<|start|>assistant"
    )

    with open(f'../datasets/{source_name}', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    out_name = "../datasets/" + source_name.removesuffix('.jsonl') + "-GPT_OSS_Safeguard.jsonl"

    done_ids = []

    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            for line in f:
                done_ids.append(json.loads(line)["ID"])

    def generate(token: str) -> None:
        for data in tqdm(dataset):
            if data["ID"] in done_ids:
                print(f'Skip {data["ID"]}')
                continue

            out_data = {}

            out_data['ID'] = data['ID']
            out_data['prompt'] = data['prompt']
            out_data['answer'] = data['answer']

            prompt = (
                f"**Question**: {data['prompt']}\n\n"
                f"**Response**: {data['answer']}"
            )

            harmony_prompt = harmony_prompt_raw.format(prompt=prompt)

            payload = {
                "model": "openai/gpt-oss-safeguard-20b",
                "prompt": harmony_prompt, 
                "temperature": 0,
                "max_tokens": 2048,
                "stop": ["<|end|>", "<|return|>", "User Prompt:", "Context:"] 
            }

            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            
            response = requests.post(f"https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            response_json = response.json()
            out_data['safeguard_evaluation'] = int(response_json['choices'][0]['text'])
            out_data['reasoning'] = response_json['choices'][0]['reasoning']

            print(f"{out_data['ID']}: {out_data['safeguard_evaluation']}")

            with open(out_name, 'a') as f:
                json_line = json.dumps(out_data)
                f.write(json_line + '\n')
        
            done_ids.append(data["ID"])
            time.sleep(2)
        print("All done!")
    
    run_with_retry(generate)


if __name__ == '__main__':
    for source_name in ('gpt-5-Baseline.jsonl', 'deepseek-chat-v3-0324-Baseline.jsonl', 'gemma-3-4b-it-Baseline.jsonl', 'gemma-3-12b-it-Baseline.jsonl', 'gemma-3-27b-it-Baseline.jsonl', 'glm-4.5-air-Baseline.jsonl', 'gpt-oss-20b-Baseline.jsonl', 'grok-4.1-fast-Baseline.jsonl', 'llama-3.3-8b-instruct-Baseline.jsonl', 'llama-3.3-70b-instruct-Baseline.jsonl', 'mistral-small-3.2-24b-instruct-Baseline.jsonl', 'qwen3-8b-Baseline.jsonl', 'qwen3-235b-a22b-Baseline.jsonl'):
        evaluate_gpt_oss_safeguard(source_name)