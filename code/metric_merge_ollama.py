from time import sleep
from ollama import chat, pull, show, list
from ollama import ChatResponse
import json
import os
from utils import *
from tqdm import tqdm
import re

eval_model = "qwen/qwen3-235b-a22b:free"
current_id = None
retries_cnt = 0

def get_prompt(prompt_file):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def get_dataset(input_file):
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:

        for line in f:
            # print("workds")
            # print(line)
            dataset.append(json.loads(line))
    return dataset


def evaluate(datafilename, eval_model, token, resume=True, refined=True, in_data=None):
    global current_id, retries_cnt

    retries_cnt = 0
    current_id = None
    
    # Check if using HuggingFace model
    use_hf = eval_model.startswith("hf:")
    
    if not use_hf:
        # For Ollama models, extract model name differently
        model_name = eval_model.replace(":","-")
    else:
        # For HuggingFace models
        model_name = eval_model.split('/')[-1]
    
    filename = datafilename.removesuffix(".jsonl").replace(
        "datasets/", f"datasets/evaluations/{model_name}/") + "_label.jsonl"
    with open('./code/IDs1000.txt', 'r') as file:
        IDs1000 = [int(line.strip()) for line in file]

    # CHANGE IF USING REGULAR PROMPTS
    S_prompt = get_prompt('./code/prompts/promptPrivacyNew.txt')
    R_prompt = get_prompt('./code/prompts/promptHarmfulNew.txt')
    I_prompt = get_prompt('./code/prompts/promptMisinformationNew.txt')

    dataset = get_dataset(datafilename)
    IDsDone = []
    if resume and os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                IDsDone.append(json.loads(line)["ID"])
    sleep_count = 0
    for data in tqdm(dataset):
        if data["ID"] not in IDs1000 or data["ID"] in IDsDone:
            print(f"Skip {data['ID']}")
            continue
        
        if retries_cnt >= 5:
            if data["ID"] != current_id:
                current_id = None
                retries_cnt = 0
            else:
                if not data["error"]:
                    data["error"] = []
                data["error"].append("Too many retries")
                IDsDone.append(data["ID"])
                retries_cnt = 0
                flag = 'a' if os.path.exists(filename) else 'w'
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, flag, encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                continue

        data["ifPrivacy"] = "idk"
        data["ifHarmful"] = "idk"
        data["ifMisinformation"] = "idk"

        if not in_data.keys() or in_data["ID"] != data["ID"]:
            print("in_data updated!")
            in_data.update(data)
            in_data["error"] = None

        data = in_data
        # sleep_count += 3
        # if sleep_count == 20:
        #     sleep(60)
        #     sleep_count = 0
        # evaluate Sensitive identity attributes
        attempts = 0
        done = False

        prompts = []

        for prompt in (S_prompt, R_prompt, I_prompt):
            # COMMENT IF USING REGULAR INSTRUCTIONS
            temp = prompt.replace("<<<QUESTION>>>", data["prompt"])
            prompts.append(temp.replace("<<<RESPONSE>>>", data["answer"]))

            # COMMENT IF USING NEW INSTRUCTIONS
            # prompt += f'\nText: "{data["answer"]}"'

        # Enough to handle json format errors; necessary when temperature>0
        while data["ifPrivacy"] == "idk" and attempts < 1:
            # print(f"Evaluating {data['answer']}")
            if use_hf:
                # Use HuggingFace model via generate_answer
                response_text, _ = generate_answer(eval_model, token, prompts[0])
                print("LLM response:", repr(response_text))
            else:
                completion = chat(
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[0]},
                    ],
                    stream=False
                )
                if completion is None:
                    continue

                response_text = completion.message.content
                # Debugging line
                print("LLM response:", repr(response_text))

            try:
                parsed_dict = extract_json_from_answer(response_text)
                data['ifPrivacy'] = parsed_dict['ifPrivacy']
                if parsed_dict['ifPrivacy'] == 'yes':
                    data['privacy'] = parsed_dict['privacy']
                    data['scorePrivacy'] = parsed_dict['scorePrivacy']
                else:
                    data['privacy'] = None
                    data['scorePrivacy'] = None
                done = True
                print('valid')
            except json.JSONDecodeError as e:
                print(
                    f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(
                        repr(response_text))
                else:
                    data['error'] = [
                        repr(completion.message.content)]
            finally:
                if not current_id:
                    current_id = data["ID"]
                    retries_cnt = 1
                else:
                    retries_cnt += 1
            attempts += 1
            if done:
                break
        # evaluate Reputation-harmful contents
        attempts = 0
        done = False
        while data["ifHarmful"] == "idk" and attempts < 1:
            if use_hf:
                # Use HuggingFace model via generate_answer
                response_text, _ = generate_answer(eval_model, token, prompts[1])
                print("LLM response:", repr(response_text))
            else:
                completion = chat(
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[1]},
                    ],
                    stream=False
                )
                if completion is None:
                    continue

                response_text = completion.message.content
                # Debugging line
                print("LLM response:", repr(response_text))

            try:
                parsed_dict = extract_json_from_answer(response_text)
                data['ifHarmful'] = parsed_dict['ifHarmful']
                if parsed_dict['ifHarmful'] == 'yes':
                    data['harmful'] = parsed_dict['harmful']
                    data['scoreHarmful'] = parsed_dict['scoreHarmful']
                else:
                    data['harmful'] = None
                    data['scoreHarmful'] = None
                done = True
                print('valid')
            except json.JSONDecodeError as e:
                print(
                    f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(
                        repr(response_text))
                else:
                    data['error'] = [
                        repr(response_text)]
            finally:
                if not current_id:
                    current_id = data["ID"]
                    retries_cnt = 1
                else:
                    retries_cnt += 1
            attempts += 1
            if done:
                break

        # evaluate Incorrect hazardous information
        attempts = 0
        done = False
        while data["ifMisinformation"] == "idk" and attempts < 1:
            if use_hf:
                # Use HuggingFace model via generate_answer
                response_text, _ = generate_answer(eval_model, token, prompts[2])
                print("LLM response:", repr(response_text))
            else:
                completion = chat(
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[2]},
                    ],
                    stream=False
                )
                if completion is None:
                    continue

                response_text = completion.message.content
                # Debugging line
                print("LLM response:", repr(response_text))

            try:
                parsed_dict = extract_json_from_answer(response_text)
                data['ifMisinformation'] = parsed_dict['ifMisinformation']
                if parsed_dict['ifMisinformation'] == 'yes':
                    data['misinformation'] = parsed_dict['misinformation']
                    data['scoreMisinformation'] = parsed_dict['scoreMisinformation']
                else:
                    data['misinformation'] = None
                    data['scoreMisinformation'] = None
                done = True
                print('valid')
            except json.JSONDecodeError as e:
                print(
                    f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(
                        repr(response_text))
                else:
                    data['error'] = [
                        repr(response_text)]
            finally:
                if not current_id:
                    current_id = data["ID"]
                    retries_cnt = 1
                else:
                    retries_cnt += 1
            attempts += 1
            if done:
                break

        # save on the fly
        flag = 'a' if os.path.exists(filename) else 'w'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, flag, encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # proxy_on()
    backlog = [
        "deepseek-chat-v3-0324_Feedback(3Iter).jsonl",
        "deepseek-chat-v3-0324-Baseline.jsonl",
        "deepseek-chat-v3-0324_Mark+Redact.jsonl",
        "deepseek-chat-v3-0324-Pre.jsonl",
        "deepseek-chat-v3-0324-PostWithEval.jsonl",
    ]

    for datafilename in backlog:
        datafilename = f"./datasets/{datafilename}"
        if not run_evaluation_with_retry(evaluate, datafilename, eval_model):
            break
        print(f"Finish: {datafilename}")
