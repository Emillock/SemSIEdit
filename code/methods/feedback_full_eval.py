"""
Feedback method with FULL EVALUATION using three separate evaluator calls.

This variant replaces the single loop_evaluator.txt call with three separate
evaluator calls using:
- promptPrivacyNew.txt (privacy detection)
- promptHarmfulNew.txt (harmful content detection)
- promptMisinformationNew.txt (misinformation detection)

The three results are combined into the same JSON structure as the original
single evaluator, maintaining compatibility with the Editor phase.
"""

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


def run_three_evaluators(
    model_name: str,
    token: str,
    question: str,
    answer: str,
    prompt_privacy_base: str,
    prompt_harmful_base: str,
    prompt_misinformation_base: str,
) -> tuple[dict, dict, dict]:
    individual_results = {}
    combined_ev = {}
    citations = {}

    # Evaluator 1: Privacy
    try:
        prompt_privacy = prompt_privacy_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer)
        ev_privacy_raw, reasoning_privacy = generate_answer(model_name, token, prompt_privacy)
        if generate_answer.last_citations:
            citations["privacy"] = generate_answer.last_citations
        ev_privacy = extract_json_from_answer(ev_privacy_raw)

        individual_results["privacy"] = {
            "reasoning": reasoning_privacy,
            "evaluation": ev_privacy
        }

        combined_ev["ifPrivacy"] = ev_privacy.get("ifPrivacy", "no")
        if combined_ev["ifPrivacy"] == "yes":
            if "privacy" in ev_privacy:
                combined_ev["privacy"] = ev_privacy["privacy"]
            if "scorePrivacy" in ev_privacy:
                combined_ev["scorePrivacy"] = ev_privacy["scorePrivacy"]
    except Exception as e:
        print(f"Error in privacy evaluator: {e}")
        combined_ev["ifPrivacy"] = "no"
        individual_results["privacy"] = {"error": str(e)}

    # Evaluator 2: Harmful
    try:
        prompt_harmful = prompt_harmful_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer)
        ev_harmful_raw, reasoning_harmful = generate_answer(model_name, token, prompt_harmful)
        if generate_answer.last_citations:
            citations["harmful"] = generate_answer.last_citations
        ev_harmful = extract_json_from_answer(ev_harmful_raw)

        individual_results["harmful"] = {
            "reasoning": reasoning_harmful,
            "evaluation": ev_harmful
        }

        combined_ev["ifHarmful"] = ev_harmful.get("ifHarmful", "no")
        if combined_ev["ifHarmful"] == "yes":
            if "harmful" in ev_harmful:
                combined_ev["harmful"] = ev_harmful["harmful"]
            if "scoreHarmful" in ev_harmful:
                combined_ev["scoreHarmful"] = ev_harmful["scoreHarmful"]
    except Exception as e:
        print(f"Error in harmful evaluator: {e}")
        combined_ev["ifHarmful"] = "no"
        individual_results["harmful"] = {"error": str(e)}

    # Evaluator 3: Misinformation
    try:
        prompt_misinformation = prompt_misinformation_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer)
        ev_misinformation_raw, reasoning_misinformation = generate_answer(model_name, token, prompt_misinformation)
        if generate_answer.last_citations:
            citations["misinformation"] = generate_answer.last_citations
        ev_misinformation = extract_json_from_answer(ev_misinformation_raw)

        individual_results["misinformation"] = {
            "reasoning": reasoning_misinformation,
            "evaluation": ev_misinformation
        }

        combined_ev["ifMisinformation"] = ev_misinformation.get("ifMisinformation", "no")
        if combined_ev["ifMisinformation"] == "yes":
            if "misinformation" in ev_misinformation:
                combined_ev["misinformation"] = ev_misinformation["misinformation"]
            if "scoreMisinformation" in ev_misinformation:
                combined_ev["scoreMisinformation"] = ev_misinformation["scoreMisinformation"]
    except Exception as e:
        print(f"Error in misinformation evaluator: {e}")
        combined_ev["ifMisinformation"] = "no"
        individual_results["misinformation"] = {"error": str(e)}

    if all("error" in v for v in individual_results.values()):
        raise Exception("All three evaluators failed")

    return combined_ev, individual_results, citations


def generate_feedback(model_name: str, source_name: str, iter_cnt: int, max_lines: int = -1) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    if model_name.startswith("hf:"):
        model_display_name = model_name.split('/')[-1]
    else:
        model_display_name = model_name.split('/')[1].removesuffix(":free") if "/" in model_name else model_name

    out_name = "datasets-DP/" + model_display_name + f"-FeedbackFullEval({iter_cnt}Iter).jsonl"
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

    with open("./code/prompts/promptPrivacyNew.txt", 'r') as f:
        prompt_eval_privacy_base = f.read()

    with open("./code/prompts/promptHarmfulNew.txt", 'r') as f:
        prompt_eval_harmful_base = f.read()

    with open("./code/prompts/promptMisinformationNew.txt", 'r') as f:
        prompt_eval_misinformation_base = f.read()

    with open("./code/prompts/loop_editor.txt", 'r') as f:
        prompt_edit_base = f.read()

    use_online = os.environ.get("USE_WEB_SEARCH", "").lower() in ("1", "true", "yes")
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
            debug_data["ID"] = data["ID"]

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

                    answer, reasoning = generate_answer(online_model, token, prompt_init)
                    if generate_answer.last_citations:
                        search_results["init"] = generate_answer.last_citations

                    print_reasoning_and_answer(reasoning, answer)

                    debug_data["init_reasoning"] = reasoning
                    debug_data["init_answer"] = answer
                    init_done = True

                for i in range(last_i, iter_cnt + 1):
                    last_i = i

                    if not eval_done:
                        print(f" Iteration #{i} - Evaluator (3 separate calls) ".center(100, '-'))
                        print()

                        ev, individual_evals, eval_citations = run_three_evaluators(
                            online_model, token, question, answer,
                            prompt_eval_privacy_base, prompt_eval_harmful_base, prompt_eval_misinformation_base,
                        )
                        if eval_citations:
                            search_results[f"it{i}_eval"] = eval_citations

                        print(" Privacy Evaluator ".center(80, '~'))
                        if "privacy" in individual_evals and "reasoning" in individual_evals["privacy"]:
                            print_reasoning_and_answer(
                                individual_evals["privacy"]["reasoning"],
                                repr(individual_evals["privacy"]["evaluation"])
                            )
                        elif "privacy" in individual_evals and "error" in individual_evals["privacy"]:
                            print(f"ERROR: {individual_evals['privacy']['error']}")
                            print()

                        print(" Harmful Evaluator ".center(80, '~'))
                        if "harmful" in individual_evals and "reasoning" in individual_evals["harmful"]:
                            print_reasoning_and_answer(
                                individual_evals["harmful"]["reasoning"],
                                repr(individual_evals["harmful"]["evaluation"])
                            )
                        elif "harmful" in individual_evals and "error" in individual_evals["harmful"]:
                            print(f"ERROR: {individual_evals['harmful']['error']}")
                            print()

                        print(" Misinformation Evaluator ".center(80, '~'))
                        if "misinformation" in individual_evals and "reasoning" in individual_evals["misinformation"]:
                            print_reasoning_and_answer(
                                individual_evals["misinformation"]["reasoning"],
                                repr(individual_evals["misinformation"]["evaluation"])
                            )
                        elif "misinformation" in individual_evals and "error" in individual_evals["misinformation"]:
                            print(f"ERROR: {individual_evals['misinformation']['error']}")
                            print()

                        print(" Combined Evaluation ".center(80, '='))
                        print(repr(ev))
                        print()

                        for eval_type in ["privacy", "harmful", "misinformation"]:
                            if eval_type in individual_evals:
                                if "reasoning" in individual_evals[eval_type]:
                                    debug_data[f"it{i}_eval_{eval_type}_reasoning"] = individual_evals[eval_type]["reasoning"]
                                    debug_data[f"it{i}_eval_{eval_type}_answer"] = repr(individual_evals[eval_type]["evaluation"])
                                elif "error" in individual_evals[eval_type]:
                                    debug_data[f"it{i}_eval_{eval_type}_error"] = individual_evals[eval_type]["error"]

                        debug_data[f"it{i}_eval_combined"] = repr(ev)

                        if ("ifPrivacy" not in ev or ev["ifPrivacy"] == "no") \
                            and ("ifHarmful" not in ev or ev["ifHarmful"] == "no") \
                            and ("ifMisinformation" not in ev or ev["ifMisinformation"] == "no"):
                            print("Iteration stopped - no SemSI detected by any evaluator")
                            break

                        eval_done = True

                    print(f" Iteration #{i} - Editor ".center(100, '-'))
                    print()

                    prompt_edit = prompt_edit_base.replace("<<<QUESTION>>>", question).replace("<<<RESPONSE>>>", answer).replace("<<<EVALUATION>>>", repr(ev))

                    answer, reasoning = generate_answer(online_model, token, prompt_edit)
                    if generate_answer.last_citations:
                        search_results[f"it{i}_edit"] = generate_answer.last_citations

                    print_reasoning_and_answer(reasoning, answer)

                    print(f" Iteration #{i} - Re-Evaluator (3 separate calls after editing) ".center(100, '-'))
                    print()

                    ev_recheck, individual_reevals, reeval_citations = run_three_evaluators(
                        online_model, token, question, answer,
                        prompt_eval_privacy_base, prompt_eval_harmful_base, prompt_eval_misinformation_base,
                    )
                    if reeval_citations:
                        search_results[f"it{i}_reeval"] = reeval_citations

                    print(" Privacy Re-Evaluator ".center(80, '~'))
                    if "privacy" in individual_reevals and "reasoning" in individual_reevals["privacy"]:
                        print_reasoning_and_answer(
                            individual_reevals["privacy"]["reasoning"],
                            repr(individual_reevals["privacy"]["evaluation"])
                        )
                    elif "privacy" in individual_reevals and "error" in individual_reevals["privacy"]:
                        print(f"ERROR: {individual_reevals['privacy']['error']}")
                        print()

                    print(" Harmful Re-Evaluator ".center(80, '~'))
                    if "harmful" in individual_reevals and "reasoning" in individual_reevals["harmful"]:
                        print_reasoning_and_answer(
                            individual_reevals["harmful"]["reasoning"],
                            repr(individual_reevals["harmful"]["evaluation"])
                        )
                    elif "harmful" in individual_reevals and "error" in individual_reevals["harmful"]:
                        print(f"ERROR: {individual_reevals['harmful']['error']}")
                        print()

                    print(" Misinformation Re-Evaluator ".center(80, '~'))
                    if "misinformation" in individual_reevals and "reasoning" in individual_reevals["misinformation"]:
                        print_reasoning_and_answer(
                            individual_reevals["misinformation"]["reasoning"],
                            repr(individual_reevals["misinformation"]["evaluation"])
                        )
                    elif "misinformation" in individual_reevals and "error" in individual_reevals["misinformation"]:
                        print(f"ERROR: {individual_reevals['misinformation']['error']}")
                        print()

                    print(" Combined Re-Evaluation ".center(80, '='))
                    print(repr(ev_recheck))
                    print()

                    for eval_type in ["privacy", "harmful", "misinformation"]:
                        if eval_type in individual_reevals:
                            if "reasoning" in individual_reevals[eval_type]:
                                debug_data[f"it{i}_reeval_{eval_type}_reasoning"] = individual_reevals[eval_type]["reasoning"]
                                debug_data[f"it{i}_reeval_{eval_type}_answer"] = repr(individual_reevals[eval_type]["evaluation"])
                            elif "error" in individual_reevals[eval_type]:
                                debug_data[f"it{i}_reeval_{eval_type}_error"] = individual_reevals[eval_type]["error"]

                    debug_data[f"it{i}_reeval_combined"] = repr(ev_recheck)

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
        print("FEEDBACK FULL EVAL GENERATION COMPLETED!")
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

    generate_feedback(model_name, "./semsi-datasets/gemini1.0-pro_label.jsonl", iter_cnt, max_lines)
