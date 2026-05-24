import os
import glob

filenames = glob.glob('./evaluations/gpt-5/*WebSearch*')

for filename in filenames:
    utility = filename.replace('evaluations', 'utility').replace('_label', '_utility')
    
    with open(filename, 'r') as f:
        print(f'{filename}: {len(f.readlines())}')
    
    if os.path.exists(utility):
        with open(utility, 'r') as f:
            print(f'{utility}: {len(f.readlines())}')
    else:
        print(f'{utility}: 0')
    
    print()
    print()
