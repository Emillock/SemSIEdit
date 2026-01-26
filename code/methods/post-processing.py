from time import sleep
from openai import OpenAI
import json
import os
from utils import *
import pandas as pd
from tqdm import tqdm

def query(opts, token, resume=True):
    # proxy_on()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=token,
    ) 
    print(opts)
    modelName = opts.model_name
    sourcefile = opts.source_file
    savedir = os.path.join("exp", sourcefile.split('/')[-2])
    os.makedirs(savedir, exist_ok=True)
    savefile = sourcefile.removesuffix("_pre.jsonl")+"_full.jsonl"
    question = opts.question
    dataFrame = pd.read_json(sourcefile, lines=True)
    line_count = 0
    # answer=None
    if resume and os.path.exists(savefile):
        with open(savefile, 'r', encoding='utf-8') as file:
            line_count = sum(1 for line in file)
    
    with open("./code/prompts/feedback_post.txt") as file:
        instructions = file.read()
        
    sleep_count = 0
    cnt=0
    for index, row in tqdm(dataFrame.iterrows(), total=dataFrame.shape[0]):
        if index < line_count:
            continue
        sleep_count += 1
        if sleep_count == 20:
            sleep(60)
            sleep_count = 0
        data = row.to_dict()
        statement = data['statement'][:-
                                      1] if data['statement'][-1] == '.' else data['statement']
        prompt = question+statement+'?'
        prompt_json = prompt
        
        ### comment if not doing post processing
        # if data["ifPrivacy"] == "no" and data["ifHarmful"] == "no" and data["ifMisinformation"] == "no":
        #     with open(savefile, 'a') as file2:
        #         json_line = json.dumps(data)
        #         file2.write(json_line + '\n')
        #     continue

        # prompt = instructions.replace('<<<TEXT>>>', data["answer"])
        # if data["ifPrivacy"] == "yes":
        #     prompt += f'\nPrivate information that can harm the subject if released:\n{data["privacy"]}'
        # if data["ifHarmful"] == "yes":
        #     prompt += f'\nInformation harmful to subject\'s reputation:\n{data["harmful"]}'
        # if data["ifMisinformation"] == "yes":
        #     prompt += f'\nIncorrect or hazardous information (misinformation):\n{data["misinformation"]}'
        ###
        
        # comment if doing without the instructions
        #prompt = instructions+prompt
        # comment if doing without the instructions sandwich
        
        prompt = instructions.replace('<<<QUESTION>>>', question+statement+'?')
        prompt = prompt.replace('<<<RESPONSE>>>', data["answer"])
        
        print("\nsend request")
        print("prompt:", prompt)


        response = client.chat.completions.create(
            model=modelName,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        print(response)
        answer = response.choices[0].message.content
        think_part_start = answer.find("</think>")
        
        if think_part_start != -1:
            think_part_end = think_part_start + len("</think>")
            answer = answer[think_part_end:].strip()

        data['prompt'] = prompt_json
        data['model'] = modelName
        data['answer'] = answer
        with open(savefile, 'a') as file2:
            json_line = json.dumps(data)
            file2.write(json_line + '\n')
    print("All done!")
    # proxy_off()


if __name__ == '__main__':
    opts = get_opts()
    run_with_retry(query, opts)
