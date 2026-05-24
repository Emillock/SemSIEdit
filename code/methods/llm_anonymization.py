import json
import os

from utils import *
from tqdm import tqdm


INFERENCE_PROMPT = """You are an expert at analyzing text for personal information leakage.

Below is a text. Your task is to identify what personal attributes of any person mentioned could be inferred from this text.

Text:
\"\"\"
{text}
\"\"\"

For each of the following attribute categories, determine if the text reveals or allows inference of such information about any person mentioned:
- Age
- Sex/Gender
- Location (city, state, country)
- Occupation/Role
- Education
- Income/Wealth
- Relationship status
- Place of birth
- Political affiliation
- Ethnicity/Race
- Health information

For each attribute where an inference IS possible, output a line in this exact format:
Type: <attribute>
Inference: <what can be inferred and from which part of the text>
Guess: <the specific guess>

If no personal attributes can be inferred for a category, skip it entirely.
Only output the inferences, nothing else."""

ANON_PROMPT = """You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous.

Below I give you a text as well as personal inferences that can be made from it. Your task is to anonymize the text to prevent these inferences, while keeping the rest of the text intact.

Text:
\"\"\"
{text}
\"\"\"

Inferences:
{inferences}

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Do not invent new information instead generalize information.

Examples of valid changes: 'my husband and I' -> 'my partner and I', 'in New York' -> 'in a major city', 'as a teacher' -> 'as a professional'."""


def parse_anonymized_output(raw_response: str) -> str:
    cleaned = raw_response
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>"):].strip()

    # Try splitting on \n# (line starting with #)
    parts = cleaned.split('\n#')
    if len(parts) >= 2:
        anonymized = '\n#'.join(parts[1:]).strip()
        # Remove leading # if the split left one
        if anonymized.startswith('#'):
            anonymized = anonymized[1:].strip()
        return anonymized

    # Fallback: find a standalone # line
    lines = cleaned.split('\n')
    for i, line in enumerate(lines):
        if line.strip() == '#':
            return '\n'.join(lines[i + 1:]).strip()

    # Final fallback: return full response
    return cleaned.strip()


def generate_llm_anonymization(model_name: str, source_name: str, max_lines: int = -1) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    if model_name.startswith("hf:"):
        model_display_name = model_name.split('/')[-1]
    else:
        model_display_name = model_name.split('/')[1].removesuffix(":free") if "/" in model_name else model_name

    out_name = "datasets-DP/" + model_display_name + "-LLMAnon.jsonl"
    log_name = out_name.replace("datasets-DP/", "logs/")

    done_ids = []

    if not os.path.exists(os.path.dirname(log_name)):
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

                # Step 1: Generate initial answer
                print(f"\n[Entry {data['ID']}] Generating initial answer...")
                initial_answer, initial_reasoning = generate_answer(model_name, token, prompt)
                print(f"Initial answer: {initial_answer[:200]}...")

                # Step 2: Inference step - identify leaking PII attributes
                print(f"[Entry {data['ID']}] Running inference step...")
                inference_prompt = INFERENCE_PROMPT.format(text=initial_answer)
                inferences_raw, inference_reasoning = generate_answer(model_name, token, inference_prompt)
                print(f"Inferences: {inferences_raw[:200]}...")

                # Step 3: Anonymization step - rewrite to prevent inferences
                inferences_stripped = inferences_raw.strip()
                if not inferences_stripped or inferences_stripped.lower().startswith("no "):
                    # No inferences found, skip anonymization
                    print(f"[Entry {data['ID']}] No inferences found, using initial answer as-is.")
                    anonymized_answer = initial_answer
                    anon_response_raw = ""
                    anon_reasoning = ""
                else:
                    print(f"[Entry {data['ID']}] Running anonymization step...")
                    anon_prompt = ANON_PROMPT.format(text=initial_answer, inferences=inferences_raw)
                    anon_response_raw, anon_reasoning = generate_answer(model_name, token, anon_prompt)
                    anonymized_answer = parse_anonymized_output(anon_response_raw)
                    print(f"Anonymized answer: {anonymized_answer[:200]}...")

                data['prompt'] = prompt
                data['model'] = model_name
                data['initial_answer'] = initial_answer
                data['answer'] = anonymized_answer
                data['inferences'] = inferences_raw

                debug_data = {
                    "initial_reasoning": initial_reasoning,
                    "initial_answer": initial_answer,
                    "inference_reasoning": inference_reasoning,
                    "inferences_raw": inferences_raw,
                    "anon_reasoning": anon_reasoning,
                    "anon_response_raw": anon_response_raw,
                    "anonymized_answer": anonymized_answer,
                }
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

    if model_name.startswith("hf:"):
        run_without_retry(generate)
    else:
        run_with_retry(generate)


if __name__ == '__main__':
    model_name = "qwen/qwen3-8b"
    max_lines = 3

    generate_llm_anonymization(model_name, "./datasets/data_src.jsonl", max_lines)
