import argparse
import datetime
import itertools
import json
import openai
import os
import re
import time
import traceback
import random
from ollama import chat

seed = os.environ.get("SHUFFLE_SEED")

tokens = os.environ.get("TOKENS", "").split(",")

def remove_reasoning(text: str) -> str:
    reasoning_end_pos = text.find("</think>")
        
    if reasoning_end_pos != -1:
        return text[(reasoning_end_pos + len("</think>")):].strip()
    
    return text

def extract_json_from_answer(text: str) -> dict:
    res = re.sub(r'```json|```', '', remove_reasoning(text)).strip()
    res = res[res.find('{'):]
    
    try:
        return json.loads(res, strict=False)
    except json.JSONDecodeError:
        print(text)
        raise

def generate_answer(model: str, token: str, prompt: str) -> tuple[str, str]:
    reasoning = ''

    if "/" in model:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=token,
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"role": "user", "content": prompt},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ]},
            ],
            stream=False,
            reasoning_effort="medium",
            extra_body={},
        )

        print(response)
    
        raw = response.choices[0].message.content
        reasoning = response.choices[0].message.reasoning or ''
    else:
        response = chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        ) 

        raw = response.message.content

    if reasoning == '':
        reasoning_end_pos = raw.find("</think>")
        
        if reasoning_end_pos != -1:
            return (raw[(reasoning_end_pos + len("</think>")):], raw[:reasoning_end_pos].removeprefix("<think>"))
    
    return (raw, reasoning)

def proxy_on():
    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"


def proxy_off():
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if "https_proxy" in os.environ:
        del os.environ["https_proxy"]


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="gemma3:4b")
    parser.add_argument('--source_file', type=str, default='./datasets/gemma-3-27b-it-Pre.jsonl')
    parser.add_argument('--question', type=str,
                        default='Why ', choices=['Why ', 'How '])
    parser.add_argument('--datafilename', type=str, default="./datasets/deepseek-chat-v3-0324_Feedback(3Iter).jsonl")

    opts = parser.parse_args()
    return opts

def run_evaluation_with_retry(func, *args, retry_delay=10, tokens=tokens, **kwargs):
    random.shuffle(tokens)
    token_cycle = itertools.cycle(tokens)
    token = next(token_cycle, None)
    print('=' * 20)
    in_data = {"ID":-1,"retries": 0}
    while token:
        print(f"Using token \"{token}\"...")
        try:
            func(*args, token, **kwargs, in_data=in_data)
            print('=' * 20)
            break
        except openai.NotFoundError as e:
            traceback.print_exception(e)
            tokens.remove(token)
            random.shuffle(tokens)
            token_cycle = itertools.cycle(tokens)
            print(f"Token \"{token}\" will not be used any more.")
        except openai.RateLimitError as e:
            traceback.print_exception(e)
            if e.body["message"].startswith("Rate limit exceeded: free-models-per-day"):
                tokens.remove(token)
                random.shuffle(tokens)
                token_cycle = itertools.cycle(tokens)
                print(f"Token \"{token}\" will not be used any more.")
            if e.body["message"].startswith("Rate limit exceeded: free-models-per-min"):
                limit_reset = int(e.body["metadata"]["headers"]["X-RateLimit-Reset"])
                now_utc = datetime.datetime.now(datetime.UTC).timestamp() * 1000
                print(f"Limit reset: {limit_reset}")
                print(f"Now: {now_utc}")
                print(f"Diff (in s): {(limit_reset - now_utc) / 1000}")
                if now_utc < limit_reset:
                    time.sleep((limit_reset - now_utc) / 1000)
        except Exception as e:
            print("Unexpected error:")
            traceback.print_exception(e)
            in_data["retries"] += 1
            # time.sleep(retry_delay)
        print("Retrying with a new token...")
        print('=' * 20)
        token = next(token_cycle, None)
    if not token:
        print("All tokens were exhausted. Stopping...")
        return False
    print('=' * 20)
    return True

def run_with_retry(func, *args, tokens=tokens, **kwargs):
    random.shuffle(tokens)
    token_cycle = itertools.cycle(tokens)
    token = next(token_cycle, None)
    print('=' * 100)
    while token:
        print(f"Using token \"{token}\"...")
        try:
            func(*args, token, **kwargs)
            print('=' * 100)
            break
        except openai.NotFoundError as e:
            traceback.print_exception(e)
            tokens.remove(token)
            random.shuffle(tokens)
            token_cycle = itertools.cycle(tokens)
            print(f"Token \"{token}\" will not be used any more.")
        except openai.RateLimitError as e:
            traceback.print_exception(e)
            if e.body["message"].startswith("Rate limit exceeded: free-models-per-day"):
                tokens.remove(token)
                random.shuffle(tokens)
                token_cycle = itertools.cycle(tokens)
                print(f"Token \"{token}\" will not be used any more.")
            if e.body["message"].startswith("Rate limit exceeded: free-models-per-min"):
                limit_reset = int(e.body["metadata"]["headers"]["X-RateLimit-Reset"])
                now_utc = datetime.datetime.now(datetime.UTC).timestamp() * 1000
                print(f"Limit reset: {limit_reset}")
                print(f"Now: {now_utc}")
                print(f"Diff (in s): {(limit_reset - now_utc) / 1000}")
                if now_utc < limit_reset:
                    time.sleep((limit_reset - now_utc) / 1000)
        except Exception as e:
            print("Unexpected error:")
            traceback.print_exception(e)
        print("Retrying with a new token...")
        print('=' * 100)
        token = next(token_cycle, None)
    if not token:
        print("All tokens were exhausted. Stopping...")
        return False
    print('=' * 100)
    return True
