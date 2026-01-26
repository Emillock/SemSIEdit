import glob
import json
import os

search_pattern = os.path.join("../../new_eval", "*.jsonl")
filenames = glob.glob(search_pattern)

for filename in filenames:
    with open(filename, "r") as read_file:
        lines = read_file.readlines()

    for line in lines:
        data = json.loads(line)
        semsi = {"ifPrivacy": data["ifPrivacy"], "privacy": data["privacy"], "scorePrivacy": data["scorePrivacy"], "ifHarmful": data["ifHarmful"], "harmful": data["harmful"], "scoreHarmful": data["scoreHarmful"], "ifMisinformation": data["ifMisinformation"], "misinformation": data["misinformation"], "scoreMisinformation": data["scoreMisinformation"]}
        if data["label"] == "false" and (data["ifPrivacy"] == "yes" or data["ifHarmful"] == "yes" or data["ifMisinformation"] == "yes"):
            print('=' * 100)
            print()
            print(f"File: {filename}")
            print()
            print(f"Question: {data['prompt']}")
            print()
            print(data["answer"])
            print()
            print("SemSI".center(100, '='))
            print()
            print(json.dumps(semsi))
            print()
            input()
    