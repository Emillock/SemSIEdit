import json

SEPARATOR_LENGTH = 150

filename1 = "./evaluations/gpt-5/gemma-3-4b-it-Feedback(3Iter)WebSearch_label.jsonl"
filename1_utility = filename1.replace("evaluations", "utility").replace("_label", "_utility")
filename2 = "./evaluations/gpt-5/gemma-3-4b-it-Feedback(3Iter)_label.jsonl"
filename2_utility = filename2.replace("evaluations", "utility").replace("_label", "_utility")

file1 = open(filename1, "r")
file1_utility = open(filename1_utility, "r")
file2 = open(filename2, "r")
file2_utility = open(filename2_utility, "r")

lines1 = file1.readlines()
lines1_utility = file1_utility.readlines()
lines2 = file2.readlines()
lines2_utility = file2_utility.readlines()

for index, tup in enumerate(zip(lines1, lines2)):
    try:
        line1, line2 = tup

        data1 = json.loads(line1)
        utility1 = json.loads(lines1_utility[index])
        data2 = json.loads(line2)
        utility2 = json.loads(lines2_utility[index])

        # if data1['label'] != 'true':
        #     continue
        
        utility_score1 = (utility1["correctness"] + utility1["relevance"] + utility1["completeness"]) / 3
        utility_score2 = (utility2["correctness"] + utility2["relevance"] + utility2["completeness"]) / 3
        
        occurrence1 = (data1['ifPrivacy'] == "yes" or data1['ifHarmful'] == "yes" or data1['ifMisinformation'] == "yes")
        occurrence2 = (data2['ifPrivacy'] == "yes" or data2['ifHarmful'] == "yes" or data2['ifMisinformation'] == "yes")
        
        # if utility_score1 < utility_score2:
        #     continue
        if utility_score1 < utility_score2 or not (occurrence1 and not occurrence2):
            continue
        
        print(f"Line #{index + 1} - {filename1}")
        print()
        print(f"Label: {data1['label']}")
        print()
        print(data1['prompt'])
        print()
        print('-' * SEPARATOR_LENGTH)
        print(data1['answer'])
        print('-' * SEPARATOR_LENGTH)
        print()
        print(f"ifPrivacy: {data1['ifPrivacy']} {f'({data1['scorePrivacy']})' if data1['ifPrivacy'] == 'yes' else ''}")
        if data1['ifPrivacy'] == "yes":
            print(data1['privacy'])
        print(f"ifHarmful: {data1['ifHarmful']} {f'({data1['scoreHarmful']})' if data1['ifHarmful'] == 'yes' else ''}")
        if data1['ifHarmful'] == "yes":
            print(data1['harmful'])
        print(f"ifMisinformation: {data1['ifMisinformation']} {f'({data1['scoreMisinformation']})' if data1['ifMisinformation'] == 'yes' else ''}")
        if data1['ifMisinformation'] == "yes":
            print(data1['misinformation'])
        print()
        print(f"Utility: {utility_score1:.2f}")
        print(f"Correctness: {utility1['correctness']}")
        print(f"Relevance: {utility1['relevance']}")
        print(f"Completeness: {utility1['completeness']}")
        print()
        print('X' * SEPARATOR_LENGTH)
        print()
        print(f"Line #{index + 1} - {filename2}")
        print()
        print(data2['prompt'])
        print()
        print('-' * SEPARATOR_LENGTH)
        print(data2['answer'])
        print('-' * SEPARATOR_LENGTH)
        print()
        print(f"ifPrivacy: {data2['ifPrivacy']} {f'({data2['scorePrivacy']})' if data2['ifPrivacy'] == 'yes' else ''}")
        if data2['ifPrivacy'] == "yes":
            print(data2['privacy'])
        print(f"ifHarmful: {data2['ifHarmful']} {f'({data2['scoreHarmful']})' if data2['ifHarmful'] == 'yes' else ''}")
        if data2['ifHarmful'] == "yes":
            print(data2['harmful'])
        print(f"ifMisinformation: {data2['ifMisinformation']} {f'({data2['scoreMisinformation']})' if data2['ifMisinformation'] == 'yes' else ''}")
        if data2['ifMisinformation'] == "yes":
            print(data2['misinformation'])
        print()
        print(f"Utility: {utility_score2:.2f}")
        print(f"Correctness: {utility2['correctness']}")
        print(f"Relevance: {utility2['relevance']}")
        print(f"Completeness: {utility2['completeness']}")
        print()
        
        input()
    except Exception:
        pass
    
    
#     if True:
#         print('=' * 50)
#         print()
#         print(f"Line #{index + 1}")
#         print()
#         print(" Without post-processing ".center(50, '='))
#         print()
#         print(f'Question: {data1["prompt"]}')
#         print()
#         print(data1["answer"])
#         print()
        
#         print(" SemSI ".center(50, '='))
#         print()
        
#         if data1["ifPrivacy"] == "yes":
#             print(f'Privacy:\n{data1["privacy"]}')
#         if data1["ifHarmful"] == "yes":
#             print(f'Harmful:\n{data1["harmful"]}')
#         if data1["ifMisinformation"] == "yes":
#             print(f'Misinformation:\n{data1["misinformation"]}')

#         print()
#         print(print(" With post-processing ".center(50, '=')))
#         print()
#         print(data2["answer"])
#         print()

#         if data2["ifPrivacy"] == "yes":
#             print(f'Privacy:\n{data2["privacy"]}')
#         if data2["ifHarmful"] == "yes":
#             print(f'Harmful:\n{data2["harmful"]}')
#         if data2["ifMisinformation"] == "yes":
#             print(f'Misinformation:\n{data2["misinformation"]}')

#         print()
#         print(" Results ".center(50, '='))
#         print()

#         if data1["ifPrivacy"] == "yes":
#             print(f'Privacy: {"eliminated" if data2["ifPrivacy"] == "no" else "not eliminated"}')
#         if data1["ifHarmful"] == "yes":
#             print(f'Harmful: {"eliminated" if data2["ifHarmful"] == "no" else "not eliminated"}')
#         if data1["ifMisinformation"] == "yes":
#             print(f'Misinformation: {"eliminated" if data2["ifMisinformation"] == "no" else "not eliminated"}')

#         print()
#         print()
#         input()
# print("Equal")