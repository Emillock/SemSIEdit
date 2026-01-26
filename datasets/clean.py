import glob
import json
import os

search_pattern = os.path.join("old_eval", "*.jsonl")
filenames = glob.glob(search_pattern)

for filename in filenames:
    read_file = open(filename, "r")
    lines = read_file.readlines()
    flag = False

    for line in lines:
        data = json.loads(line)
        data["error"] = None
        
        mode = "a" if flag else "w"
        with open(filename, mode, encoding="utf-8") as write_file:
            write_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            flag = True
    