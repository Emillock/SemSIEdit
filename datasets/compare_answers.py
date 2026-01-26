import sys
import json
import math

filename1 = "./gemma3-4b-Baseline.jsonl"
filename2 = "./gemma3-4b-Feedback(3Iter).jsonl"

file1 = open(filename1, "r")
file2 = open(filename2, "r")

lines1 = file1.readlines()
lines2 = file2.readlines()

for index, tup in enumerate(zip(lines1, lines2)):
    line1, line2 = tup

    data1 = json.loads(line1)
    data2 = json.loads(line2)
    
    print(f"Line #{index + 1} - {filename1}")
    print()
    print(data1['prompt'])
    print()
    print(data1['answer'])
    print()
    print(f"ifPrivacy: {data1['ifPrivacy']}")
    print(f"ifHarmful: {data1['ifHarmful']}")
    print(f"ifMisinformation: {data1['ifMisinformation']}")
    print()
    print(f"Line #{index + 1} - {filename2}")
    print()
    print(data2['prompt'])
    print()
    print(data2['answer'])
    print()
    print(f"ifPrivacy: {data2['ifPrivacy']}")
    print(f"ifHarmful: {data2['ifHarmful']}")
    print(f"ifMisinformation: {data2['ifMisinformation']}")
    print()
    
    input()
    
    
    # if True:
    #     print('=' * 50)
    #     print()
    #     print(f"Line #{index + 1}")
    #     print()
    #     print(" Without post-processing ".center(50, '='))
    #     print()
    #     print(f'Question: {data1["prompt"]}')
    #     print()
    #     print(data1["answer"])
    #     print()
        
    #     print(" SemSI ".center(50, '='))
    #     print()
        
    #     if data1["ifPrivacy"] == "yes":
    #         print(f'Privacy:\n{data1["privacy"]}')
    #     if data1["ifHarmful"] == "yes":
    #         print(f'Harmful:\n{data1["harmful"]}')
    #     if data1["ifMisinformation"] == "yes":
    #         print(f'Misinformation:\n{data1["misinformation"]}')

    #     print()
    #     print(print(" With post-processing ".center(50, '=')))
    #     print()
    #     print(data2["answer"])
    #     print()

    #     if data2["ifPrivacy"] == "yes":
    #         print(f'Privacy:\n{data2["privacy"]}')
    #     if data2["ifHarmful"] == "yes":
    #         print(f'Harmful:\n{data2["harmful"]}')
    #     if data2["ifMisinformation"] == "yes":
    #         print(f'Misinformation:\n{data2["misinformation"]}')

    #     print()
    #     print(" Results ".center(50, '='))
    #     print()

    #     if data1["ifPrivacy"] == "yes":
    #         print(f'Privacy: {"eliminated" if data2["ifPrivacy"] == "no" else "not eliminated"}')
    #     if data1["ifHarmful"] == "yes":
    #         print(f'Harmful: {"eliminated" if data2["ifHarmful"] == "no" else "not eliminated"}')
    #     if data1["ifMisinformation"] == "yes":
    #         print(f'Misinformation: {"eliminated" if data2["ifMisinformation"] == "no" else "not eliminated"}')

    #     print()
    #     print()
    #     input()
print("Equal")