import json
import re
from utils import *
from openai import OpenAI
from tqdm import tqdm

def get_prompt(prompt_file: str) -> str:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()

def get_dataset(input_file: str) -> list[dict]:
    dataset = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def evaluate_utility(datafilename: str, eval_model: str, token: str, in_data: None) -> None:
    print(eval_model)
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=token
    )
    
    model_name = eval_model.removesuffix(":free").split('/', 1)[1]
    filename = datafilename.removesuffix(".jsonl").replace("datasets/", f"datasets/utility/{model_name}/") + "_utility.jsonl"
    
    utility_prompt_base = get_prompt("./code/prompts/utilityPrompt.txt")
    dataset = get_dataset(datafilename)

    IDsDone = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                IDsDone.append(json.loads(line)["ID"])
    
    done_feedback_rows = [None] * len(dataset)
    if "Feedback" in filename:
        if "OnlyPre" in filename:
            feedback_num = 0
        if "1Iter" in filename:
            feedback_num = 1
        if "2Iter" in filename:
            feedback_num = 2
        if "3Iter" in filename:
            feedback_num = 3
        
        semsi_dataset_filename = filename.replace('utility', 'evaluations', 1).replace('utility', 'label')
        semsi_dataset = get_dataset(semsi_dataset_filename)
        
        for i in range(feedback_num + 1, 4):
            ith_iter_util_dataset_filename =  re.sub(r'\(.*\)', f'({i}Iter)', filename)
            ith_iter_semsi_dataset_filename = re.sub(r'\(.*\)', f'({i}Iter)', semsi_dataset_filename)
            if os.path.exists(ith_iter_semsi_dataset_filename):
                ith_iter_util_dataset = get_dataset(ith_iter_util_dataset_filename)
                ith_iter_semsi_dataset = get_dataset(ith_iter_semsi_dataset_filename)
                for i, (ith_iter_row, dataset_row) in enumerate(zip(ith_iter_semsi_dataset, semsi_dataset)):
                    if done_feedback_rows[i] is None and ith_iter_row['answer'] == dataset_row['answer']:
                        done_feedback_rows[i] = ith_iter_util_dataset[i]

    for i, data in enumerate(tqdm(dataset)):
        if data["ID"] in IDsDone:
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

        if in_data.get("ID") != data["ID"]:
            print("in_data updated!")
            in_data.update(data)
            in_data["error"] = None
        
        data = in_data
        
        utility_prompt = utility_prompt_base.replace("<<<QUERY>>>", data["prompt"])
        utility_prompt = utility_prompt.replace("<<<RESPONSE>>>", data["answer"])
        completion = client.chat.completions.create(
            model=eval_model,
            messages=[
                {"role": "user", "content": utility_prompt}
            ],
            stream=False,
            reasoning_effort='medium',
        )
        if completion is None or not completion.choices:
            continue

        print(len(completion.choices))
        
        print("LLM response:", repr(completion.choices[0].message.content))
        
        parsed_dict = extract_json_from_answer(completion.choices[0].message.content)
        parsed_dict["ID"] = data["ID"]
        parsed_dict["label"] = data["label"]

        data.pop("retries", None)
        in_data["retries"] = 0
        
        flag = 'a' if os.path.exists(filename) else 'w'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, flag, encoding='utf-8') as f:
            f.write(json.dumps(parsed_dict, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    opts = get_opts()
    run_evaluation_with_retry(evaluate_utility, opts.datafilename)
    print(f"Finish: {opts.datafilename}")
