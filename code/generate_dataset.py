import os
import re
import sys

from methods.baseline import generate_baseline
from methods.feedback import generate_feedback
from methods.llm_anonymization import generate_llm_anonymization
from methods.critic import generate_critic
from metric_merge import evaluate as evaluate_semsi
from multiprocessing import Process
from utility_evaluation import evaluate_utility
from utils import run_evaluation_with_retry, run_with_retry

DATA_SOURCE = "./semsi-datasets/gemini1.0-pro_label.jsonl"
# DATA_SOURCE = "./datasets/data_src.jsonl"
FEEDBACK_ARG_PATTERN = re.compile(r'(-f|--feedback)(\d+)')
CRITIC_ARG_PATTERN = re.compile(r'(-c|--critic)(\d+)')
EVAL_MODEL = "openai/gpt-5"
# EVAL_MODEL = "openai/gpt-oss-20b:free"
# EVAL_MODEL = "qwen/qwen3-235b-a22b:free"

NO_REASONING = os.environ.get("NO_REASONING", "").lower() in ("1", "true", "yes")
REASONING_ENABLED = False if NO_REASONING else None

def evaluate_full(datafilename: str, eval_model: str) -> None:
    datafilename = f"./datasets-DP/{datafilename}"
    run_evaluation_with_retry(evaluate_semsi, datafilename, eval_model)
    run_evaluation_with_retry(evaluate_utility, datafilename, eval_model)

if __name__ == "__main__":
    # for model in ('qwen/qwen3-8b', 'openai/gpt-5', 'deepseek/deepseek-chat-v3-0324', 'z-ai/glm-4.5-air', 'google/gemma-3-4b-it', 'google/gemma-3-12b-it', 'google/gemma-3-27b-it', 'openai/gpt-5', 'openai/gpt-oss-20b', 'x-ai/grok-4.1-fast', 'meta-llama/llama-3.3-70b-instruct', 'mistralai/mistral-small-3.2-24b-instruct', 'qwen/qwen3-235b-a22b', 'meta-llama/llama-3.3-8b-instruct'):
    #     evaluate_full(model.split('/')[1] + '-Feedback(OnlyPre).jsonl', EVAL_MODEL)
    # sys.exit(0)

    if len(sys.argv) < 4:
        print(f"Usage: python {sys.argv[0]} [lines] <(-f[iter_cnt] | --feedback[iter_cnt]) [models]> <(-b | --baseline) [models]> <(-a | --anon) [models]> <(-c[iter_cnt] | --critic[iter_cnt]) [models]>")
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
            critic_arg_match = CRITIC_ARG_PATTERN.fullmatch(arg)
            if feedback_arg_match:
                cur_method = "feedback"
                iter_cnt = int(feedback_arg_match.group(2))
            elif critic_arg_match:
                cur_method = "critic"
                iter_cnt = int(critic_arg_match.group(2))
            elif arg == '-b' or arg == '--baseline':
                cur_method = "baseline"
            elif arg == '-a' or arg == '--anon':
                cur_method = "anon"
            elif arg == '-e' or arg == '--eval':
                cur_method = "eval"
            elif cur_method and not arg.startswith('-'):
                if cur_method == "baseline":
                    func = generate_baseline
                    args = (arg, DATA_SOURCE, lines)
                    kwargs = {"reasoning_enabled": REASONING_ENABLED}
                elif cur_method == "feedback":
                    func = generate_feedback
                    args = (arg, DATA_SOURCE, iter_cnt, lines)
                    kwargs = {"reasoning_enabled": REASONING_ENABLED}
                elif cur_method == "critic":
                    func = generate_critic
                    args = (arg, DATA_SOURCE, iter_cnt, 5, lines)  # 5 = max_interaction
                    kwargs = {}
                elif cur_method == "anon":
                    func = generate_llm_anonymization
                    args = (arg, DATA_SOURCE, lines)
                    kwargs = {}
                elif cur_method == "eval":
                    func = evaluate_full
                    args = (arg, EVAL_MODEL)
                    kwargs = {}
                p = Process(target=func, args=args, kwargs=kwargs)
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
