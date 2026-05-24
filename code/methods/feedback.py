import json
import os

from openai import PermissionDeniedError
from tqdm import tqdm
from utils import extract_json_from_answer, generate_answer, run_with_retry, run_without_retry


def print_reasoning_and_answer(reasoning: str, answer: str) -> None:
    print(" Reasoning ".center(50, '-'))
    print()
    print("(None)" if not reasoning else reasoning)
    print()
    print(" Answer ".center(50, '-'))
    print()
    print(answer)
    print()


def generate_feedback(model_name: str, source_name: str, iter_cnt: int, max_lines: int = -1,
                      reasoning_enabled: bool = None) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    if model_name.startswith("hf:"):
        model_display_name = model_name.split('/')[-1]
    else:
        model_display_name = model_name.split('/')[1].removesuffix(":free") if "/" in model_name else model_name

    use_online = os.environ.get("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes")
    web_suffix = "WebSearch" if use_online else ""
    no_reasoning_suffix = "-NoReasoning" if reasoning_enabled is False else ""

    out_name = "datasets-DP/" + model_display_name + f"-Feedback({iter_cnt}Iter){web_suffix}{no_reasoning_suffix}.jsonl"
    log_name = out_name.replace("datasets-DP/", "logs/")

    if not os.path.exists(log_name):
        os.makedirs(os.path.dirname(log_name), exist_ok=True)

    done_ids = []
    if os.path.exists(out_name):
        with open(out_name, 'r') as f:
            for line in f:
                done_ids.append(json.loads(line)["ID"])

    with open("./code/prompts/initial_pre.txt", 'r') as f:
        prompt_init_base = f.read()

    with open("./code/prompts/loop_evaluator.txt", 'r') as f:
        prompt_eval_base = f.read()

    with open("./code/prompts/loop_editor.txt", 'r') as f:
        prompt_edit_base = f.read()
    online_model = model_name + ":online" if use_online else model_name
    if use_online:
        print("Web search enabled via OpenRouter :online suffix")

    init_done = eval_done = False
    last_i = 1
    debug_data = {}
    answer = ""
    ev = {}

    def generate(token: str) -> None:
        nonlocal init_done, eval_done, last_i, debug_data, answer, ev

        for data in tqdm(dataset):
            if data["ID"] in done_ids:
                print(f'Skip {data["ID"]}')
                continue

            statement = data['statement'][:-1] if data['statement'][-1] == '.' else data['statement']
            question = f"Why {statement}?"
            prompt_json = question

            data["prompt"] = prompt_json
            data["model"] = model_name

            print(f"\n{'='*60}")
            print(f"PROCESSING ENTRY ID: {data['ID']}")
            print(f"Statement: {statement}")
            print(f"{'='*60}")

            try:
                search_results = {}

                if not init_done:
                    prompt_init = prompt_init_base.replace("<<<QUESTION>>>", question)

                    print(f"Question: {question}")
                    print()
                    print(" Initial prompt ".center(100, '-'))
                    print()

                    answer, reasoning = generate_answer(online_model, token, prompt_init,
                                                        reasoning_enabled=reasoning_enabled)
                    if generate_answer.last_citations:
                        search_results["init"] = generate_answer.last_citations

                    print_reasoning_and_answer(reasoning, answer)

                    debug_data["init_reasoning"] = reasoning
                    debug_data["init_answer"] = answer
                    init_done = True

                for i in range(last_i, iter_cnt + 1):
                    last_i = i

                    if not eval_done:
                        print(f" Iteration #{i} - Evaluator ".center(100, '-'))
                        print()

                        prompt_eval = prompt_eval_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer)

                        ev_raw, reasoning = generate_answer(online_model, token, prompt_eval,
                                                            reasoning_enabled=reasoning_enabled)
                        if generate_answer.last_citations:
                            search_results[f"it{i}_eval"] = generate_answer.last_citations

                        ev = extract_json_from_answer(ev_raw)

                        print_reasoning_and_answer(reasoning, repr(ev))

                        debug_data[f"it{i}_eval_reasoning"] = reasoning
                        debug_data[f"it{i}_eval_answer"] = repr(ev)

                        if ("ifPrivacy" not in ev or ev["ifPrivacy"] == "no") \
                            and ("ifHarmful" not in ev or ev["ifHarmful"] == "no") \
                            and ("ifMisinformation" not in ev or ev["ifMisinformation"] == "no"):
                            print("Iteration stopped - no SemSI")
                            break

                        eval_done = True

                    print(f" Iteration #{i} - Editor ".center(100, '-'))
                    print()

                    prompt_edit = prompt_edit_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer).replace("<<<EVALUATION>>>", repr(ev))

                    answer, reasoning = generate_answer(online_model, token, prompt_edit,
                                                        reasoning_enabled=reasoning_enabled)
                    if generate_answer.last_citations:
                        search_results[f"it{i}_edit"] = generate_answer.last_citations

                    print_reasoning_and_answer(reasoning, answer)

                    print(f" Iteration #{i} - Re-Evaluator (after editing) ".center(100, '-'))
                    print()

                    prompt_eval_recheck = prompt_eval_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer)

                    ev_raw_recheck, reasoning_recheck = generate_answer(online_model, token, prompt_eval_recheck,
                                                                        reasoning_enabled=reasoning_enabled)
                    if generate_answer.last_citations:
                        search_results[f"it{i}_reeval"] = generate_answer.last_citations
                    ev_recheck = extract_json_from_answer(ev_raw_recheck)

                    print_reasoning_and_answer(reasoning_recheck, repr(ev_recheck))

                    debug_data[f"it{i}_reeval_reasoning"] = reasoning_recheck
                    debug_data[f"it{i}_reeval_answer"] = repr(ev_recheck)

                    if ("ifPrivacy" not in ev_recheck or ev_recheck["ifPrivacy"] == "no") \
                        and ("ifHarmful" not in ev_recheck or ev_recheck["ifHarmful"] == "no") \
                        and ("ifMisinformation" not in ev_recheck or ev_recheck["ifMisinformation"] == "no"):
                        print("Iteration stopped after re-evaluation - no SemSI")
                        eval_done = False
                        break
                    else:
                        ev = ev_recheck
                        eval_done = True

                    debug_data[f"it{i}_edit_answer"] = answer

                data["answer"] = answer
                if search_results:
                    data["search_results"] = search_results

                print(f"\n{'='*60}")
                print(f"ENTRY {data['ID']} SUMMARY:")
                print(f"   Final answer length: {len(answer)}")
                print(f"{'='*60}\n")
            except PermissionDeniedError as e:
                msg = f"Permission denied: {e.body['message']}"
                data["error"] = debug_data["error"] = [msg]
                print(msg)

            with open(out_name, 'a') as f:
                json_line = json.dumps(data)
                f.write(json_line + '\n')

            init_done = eval_done = False
            last_i = 1
            done_ids.append(data["ID"])

            with open(log_name, 'a', encoding='utf-8') as f:
                f.write(json.dumps(debug_data, ensure_ascii=False) + '\n')

            debug_data = {}

        print('=' * 100)
        print("FEEDBACK GENERATION COMPLETED!")
        print('=' * 100)
        print(f"   Total entries processed: {len(done_ids)}")
        print('=' * 100)
        return

    if model_name.startswith("hf:"):
        run_without_retry(generate)
    else:
        run_with_retry(generate)

if __name__ == "__main__":
    iter_cnt = 3
    model_name = "qwen/qwen3-235b-a22b:free"
    max_lines = 100

    # Set NO_REASONING=1 to disable thinking on hybrid-reasoning models
    # (qwen3-235b-a22b, qwen3-8b, glm-4.5-air, grok-4.1-fast).
    no_reasoning = os.environ.get("NO_REASONING", "").lower() in ("1", "true", "yes")
    reasoning_enabled = False if no_reasoning else None

    generate_feedback(model_name, "./semsi-datasets/gemini1.0-pro_label.jsonl", iter_cnt, max_lines,
                      reasoning_enabled=reasoning_enabled)
