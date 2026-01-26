import json
import os

from openai import PermissionDeniedError
from tqdm import tqdm
from utils import extract_json_from_answer, generate_answer, run_with_retry

def print_reasoning_and_answer(reasoning: str, answer: str) -> None:
    print(" Reasoning ".center(50, '-'))
    print()
    print("(None)" if not reasoning else reasoning)
    print()
    print(" Answer ".center(50, '-'))
    print()
    print(answer)
    print()

def generate_feedback(model_name: str, source_name: str, iter_cnt: int, max_lines: int = -1) -> None:
    with open(source_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if max_lines != -1:
            lines = lines[:max_lines]
        dataset = [json.loads(line) for line in lines]

    out_name = "datasets/" + model_name.split('/')[1].removesuffix(":free") + f"-Feedback({iter_cnt}Iter).jsonl"
    log_name = out_name.replace("datasets/", "logs/")
    
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
            
            try:
                if not init_done:
                    prompt_init = prompt_init_base.replace("<<<QUESTION>>>", question)
                    
                    print(f"Question: {question}")
                    print()
                    print(" Initial prompt ".center(100, '-'))
                    print()
                    
                    answer, reasoning = generate_answer(model_name, token, prompt_init)
                    
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

                        ev_raw, reasoning = generate_answer(model_name, token, prompt_eval)

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
                    
                    answer, reasoning = generate_answer(model_name, token, prompt_edit)

                    print_reasoning_and_answer(reasoning, answer)
                    
                    eval_done = False
                    
                    debug_data[f"it{i}_edit_reasoning"] = reasoning
                    debug_data[f"it{i}_edit_answer"] = answer
                
                data["answer"] = answer
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
        print("Done!")
        return
    
    run_with_retry(generate)

if __name__ == "__main__":
    iter_cnt = 3
    model_name = "qwen/qwen3-235b-a22b:free"
    max_lines = 100
    
    generate_feedback(model_name, "./semsi-datasets/gemini1.0-pro_label.jsonl", iter_cnt, max_lines)
