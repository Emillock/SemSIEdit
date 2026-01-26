import json
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python check_answers.py [file] [0/1]")
        return
    
    filename = sys.argv[1]
    mode = bool(int(sys.argv[2]))
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        data = json.loads(line)
        
        if mode or data["ifPrivacy"] == "yes" or data["ifHarmful"] == "yes" or data["ifMisinformation"] == "yes":
            print('=' * 100)
            print()
            print(f"Line #{index + 1}")
            print()
            print(f"Question: {data['prompt']}")
            print()
            print(f'LLM Response:\n\n{data['answer']}')
            print()
            print('-' * 100)
            print()
            if data["ifPrivacy"] == "yes":
                print(f'Privacy (toxicity: {data["scorePrivacy"]}):\n\n{data["privacy"]}')
                print()
            if data["ifHarmful"] == "yes":
                print(f'Harmful (toxicity: {data["scoreHarmful"]}):\n\n{data["harmful"]}')
                print()
            if data["ifMisinformation"] == "yes":
                print(f'Misinformation (toxicity: {data["scoreMisinformation"]}):\n\n{data["misinformation"]}')
                print()
            print('=' * 100)
            print()
            input()

if __name__ == "__main__":
    main()
