import os
import glob

filenames = glob.glob('*WebSearch*')

for filename in filenames:
    # utility = filename.replace('evaluations', 'utility').replace('_label', '_utility')
    
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            print(f'{filename}: {len(f.readlines())}')
    
        # if os.path.exists(utility):
        #     with open(utility, 'r') as f:
        #         print(f'{utility}: {len(f.readlines())}')
        # else:
        #     print(f'{utility}: 0')
        
        print()
        print()
