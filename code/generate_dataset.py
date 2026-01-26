import re
import sys

from methods.baseline import generate_baseline
from methods.feedback import generate_feedback
from metric_merge import evaluate as evaluate_semsi
from multiprocessing import Process
from utility_evaluation import evaluate_utility
from utils import run_evaluation_with_retry, run_with_retry

DATA_SOURCE = "./semsi-datasets/gemini1.0-pro_label.jsonl"
FEEDBACK_ARG_PATTERN = re.compile(r'(-f|--feedback)(\d+)')
EVAL_MODEL = "qwen/qwen3-8b"
# EVAL_MODEL = "qwen/qwen3-235b-a22b:free"

def evaluate_full(datafilename: str) -> None:   
    datafilename = f"./datasets/{datafilename}"
    run_evaluation_with_retry(evaluate_semsi, datafilename, EVAL_MODEL)
    run_evaluation_with_retry(evaluate_utility, datafilename, EVAL_MODEL)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} [lines] <(-f[iter_cnt] | --feedback[iter_cnt]) [models]> <(-b | --baseline) [models]>")
        sys.exit()

    lines = int(sys.argv[1])
    
    if lines < 1:
        print("The number of lines must be greater than 1.")
        sys.exit()

    processes = []

    cur_method = None
    iter_cnt = 0
    try:
        for arg in sys.argv[2:]:
            feedback_arg_match = FEEDBACK_ARG_PATTERN.fullmatch(arg)
            if feedback_arg_match:
                cur_method = "feedback"
                iter_cnt = int(feedback_arg_match.group(2))
            elif arg == '-b' or arg == '--baseline':
                cur_method = "baseline"
            elif arg == '-e' or arg == '--eval':
                cur_method = "eval"
            elif cur_method and not arg.startswith('-'):
                if cur_method == "baseline":
                    func = generate_baseline
                    args = (arg, DATA_SOURCE, lines)
                elif cur_method == "feedback":
                    func = generate_feedback
                    args = (arg, DATA_SOURCE, iter_cnt, lines)
                elif cur_method == "eval":
                    func = evaluate_full
                    args = (arg,)
                p = Process(target=func, args=args)
                p.start()
                processes.append(p)
            else:
                print(f"Usage: python {sys.argv[0]} <(-f[iter_cnt] | --feedback[iter_cnt]) [models]> <(-b | --baseline) [models]>")
                sys.exit()
        
        for p in processes:
            p.join()
        
        print("All processes have finished.")
    except KeyboardInterrupt:
        for p in processes:
            p.kill()
            p.join()
