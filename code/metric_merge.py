from time import sleep
from openai import OpenAI, RateLimitError
import json
import os
from utils import *
from tqdm import tqdm
import re
import sys

# eval_model = "qwen/qwen3-235b-a22b:free"

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
    # Check if using HuggingFace model
    use_hf = eval_model.startswith("hf:")
    
    if not use_hf:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=token
        )

    model_name = eval_model.removesuffix(":free").split('/', 1)[1] if "/" in eval_model else eval_model.split(':')[-1]
    filename = datafilename.removesuffix(".jsonl").replace("datasets/", f"datasets/evaluations/{model_name}/") + "_label.jsonl"
    with open('./code/IDs1000.txt', 'r') as file:
        IDs1000 = [int(line.strip()) for line in file]

    # CHANGE IF USING REGULAR PROMPTS
    S_prompt = get_prompt('./code/prompts/promptPrivacyNew.txt')
    R_prompt = get_prompt('./code/prompts/promptHarmfulNew.txt')
    I_prompt = get_prompt('./code/prompts/promptMisinformationNew.txt')

    dataset = get_dataset(datafilename)
    IDsDone = []

    done_feedback_rows = [None] * len(dataset)
    if "Feedback" in filename:
        if "OnlyPre" in filename or "0Iter" in filename:
            feedback_num = 0
        elif "1Iter" in filename:
            feedback_num = 1
        elif "2Iter" in filename:
            feedback_num = 2
        elif "3Iter" in filename:
            feedback_num = 3
        
        for i in range(feedback_num + 1, 4):
            ith_iter_fb_filename = re.sub(r'\(.*\)', f'({i}Iter)', filename)
            if os.path.exists(ith_iter_fb_filename):
                ith_iter_fb_dataset = get_dataset(ith_iter_fb_filename)
                for i, (ith_iter_row, dataset_row) in enumerate(zip(ith_iter_fb_dataset, dataset)):
                    if done_feedback_rows[i] is None and ith_iter_row['answer'] == dataset_row['answer']:
                        done_feedback_rows[i] = ith_iter_row
    print([i for i, _ in enumerate(done_feedback_rows) if _])

    if resume and os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                IDsDone.append(json.loads(line)["ID"])

    for i, data in enumerate(tqdm(dataset)):
        if data["ID"] not in IDs1000 or data["ID"] in IDsDone:
            print(f"Skip {data['ID']}")
            continue

        if done_feedback_rows[i] is not None:
            print('Row already evaluated in other feedback file')
            data = done_feedback_rows[i]
            IDsDone.append(data["ID"])
            in_data["retries"] = 0
            flag = 'a' if os.path.exists(filename) else 'w'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, flag, encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            continue
        
        if in_data["retries"] >= 5:
            if not data.get("error"):
                    data["error"] = []
            data["error"].append("Too many retries")
            IDsDone.append(data["ID"])
            in_data["retries"] = 0
            flag = 'a' if os.path.exists(filename) else 'w'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, flag, encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            continue
        
        data["ifPrivacy"] = "idk"
        data["ifHarmful"] = "idk"
        data["ifMisinformation"] = "idk"

        if in_data.get("ID") != data["ID"]:
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
            #prompt += f'\nText: "{data["answer"]}"'
        

        while data["ifPrivacy"] == "idk" and attempts < 1:  # Enough to handle json format errors; necessary when temperature>0
            # print(f"Evaluating {data['answer']}")
            if use_hf:
                # Use HuggingFace model via generate_answer
                response_text, _ = generate_answer(eval_model, token, prompts[0])
                print("LLM response:", repr(response_text))  # Debugging line
            else:
                completion = client.chat.completions.create(
                    # temperature=0,
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[0]}
                    ],
                    stream=False,
                    reasoning_effort="medium"
                )
                if completion is None or not completion.choices:
                    continue
                
                response_text = completion.choices[0].message.content
                print("LLM response:", repr(response_text))  # Debugging line
            

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
                print(f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(repr(response_text))
                else:
                    data['error'] = [repr(response_text)]

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
                completion = client.chat.completions.create(
                    temperature=0,
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[1]}
                    ],
                    stream=False,
                    reasoning_effort="medium"
                )
                if completion is None or not completion.choices:
                    continue
                response_text = completion.choices[0].message.content
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
                print(f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(repr(response_text))
                else:
                    data['error'] = [repr(response_text)]

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
                completion = client.chat.completions.create(
                    temperature=0,
                    model=eval_model,
                    messages=[
                        {"role": "user", "content": prompts[2]}
                    ],
                    stream=False,
                    reasoning_effort="medium"
                )
                if completion is None or not completion.choices:
                    continue
                response_text = completion.choices[0].message.content
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
                print(f"error: {str(e)}, error answer: {repr(response_text)}")
                if 'error' in data:
                    data['error'].append(repr(response_text))
                else:
                    data['error'] = [repr(response_text)]

            attempts += 1
            if done:
                break

        data.pop("retries", None)
        in_data["retries"] = 0
        # save on the fly
        flag = 'a' if os.path.exists(filename) else 'w'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, flag, encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # proxy_on()
    backlog = ['qwen3-235b-a22b-Pre.jsonl', 'gpt-5-Feedback(3Iter).jsonl', 'glm-4.5-air-Feedback(2Iter).jsonl', 'deepseek-chat-v3-0324-Feedback(2Iter).jsonl', 'glm-4.5-air-Feedback(1Iter).jsonl', 'llama-3.3-70b-instruct-Feedback(3Iter).jsonl', 'deepseek-chat-v3-0324-PostWithEval.jsonl', 'qwen3-8b-Baseline.jsonl', 'gemma-3-4b-it-Feedback(2Iter).jsonl', 'deepseek-chat-v3-0324-Mark+Redact.jsonl', 'deepseek-chat-v3-0324-Feedback(1Iter).jsonl', 'gpt-oss-20b-Feedback(2Iter).jsonl', 'gemma3-4b-Feedback(2Iter).jsonl', 'gemma3-4b-Feedback(3Iter).jsonl', 'gemma-3-27b-it-Feedback(3Iter).jsonl', 'mistral-small-3.2-24b-instruct-Baseline.jsonl', 'deepseek-chat-v3-0324-Pre.jsonl', 'gemma-3-4b-it-Feedback(3Iter).jsonl', 'qwen3-235b-a22b-Feedback(3Iter).jsonl', 'qwen3-235b-a22b-Baseline.jsonl', 'llama-3.3-8b-instruct-Feedback(2Iter).jsonl', 'gemma-3-27b-it-Feedback(1Iter).jsonl', 'llama-3.3-8b-instruct-Baseline.jsonl', 'gemma-3-27b-it-Feedback(2Iter).jsonl', 'glm-4.5-air-Baseline.jsonl', 'qwen3-235b-a22b-Feedback(1Iter).jsonl', 'qwen3-235b-a22b-Pre(S).jsonl', 'gemma-3-27b-it-Baseline.jsonl', 'gpt-oss-20b-Feedback(1Iter).jsonl', 'mistral-small-3.2-24b-instruct-Pre(S).jsonl', 'qwen3-235b-a22b-Feedback(2Iter).jsonl', 'gemma-3-4b-it-Feedback(1Iter).jsonl', 'mistral-small-3.2-24b-instruct-Feedback(2Iter).jsonl', 'gemma-3-27b-it-Pre.jsonl', 'gemma3-4b-Baseline.jsonl', 'gemma-3-4b-it-Baseline.jsonl', 'gemma3-4b-Feedback(1Iter).jsonl', 'mistral-small-3.2-24b-instruct-Feedback(1Iter).jsonl', 'llama-3.3-8b-instruct-Feedback(1Iter).jsonl', 'gemma-3-12b-it-Feedback(3Iter).jsonl', 'gemma-3-12b-it-Feedback(1Iter).jsonl', 'gemma-3-12b-it-Baseline.jsonl', 'glm-4.5-air-Feedback(3Iter).jsonl', 'gpt-5-Baseline.jsonl', 'gpt-oss-20b-Baseline.jsonl', 'llama-3.3-70b-instruct-Feedback(1Iter).jsonl', 'qwen3-235b-a22b-Mark+Redact.jsonl', 'llama-3.3-70b-instruct-Feedback(2Iter).jsonl', 'deepseek-chat-v3-0324-Feedback(3Iter).jsonl', 'gemma-3-12b-it-Feedback(2Iter).jsonl', 'gpt-5-Feedback(2Iter).jsonl', 'llama-3.3-70b-instruct-Baseline.jsonl', 'llama-3.3-8b-instruct-Feedback(3Iter).jsonl', 'qwen3-8b-Feedback(3Iter).jsonl', 'gpt-oss-20b-Feedback(3Iter).jsonl', 'mistral-small-3.2-24b-instruct-Feedback(3Iter).jsonl', 'gpt-5-Feedback(1Iter).jsonl', 'mistral-small-3.2-24b-instruct-Pre.jsonl', 'deepseek-chat-v3-0324-Baseline.jsonl']

    eval_model = sys.argv[1]

    for datafilename in backlog:
        datafilename = f"./datasets/{datafilename}"
        if not run_evaluation_with_retry(evaluate, datafilename, eval_model):
            break
        print(f"Finish: {datafilename}")
