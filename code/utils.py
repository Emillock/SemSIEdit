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
try:
    from ollama import chat
except ImportError:
    chat = None

try:
    from transformers import pipeline, BitsAndBytesConfig
    import torch
except ImportError:
    pipeline = None
    BitsAndBytesConfig = None
    torch = None

from dotenv import load_dotenv

# Load .env regardless of where the script is invoked from. utils.py lives in code/,
# so this resolves to code/.env even when CWD is the project root.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

seed = os.environ.get("SHUFFLE_SEED")

tokens = os.environ.get("TOKENS", "").split(",")

pipe = None

def remove_reasoning(text: str) -> str:
    reasoning_end_pos = text.find("</think>")
        
    if reasoning_end_pos != -1:
        return text[(reasoning_end_pos + len("</think>")):].strip()
    
    return text

def extract_json_from_answer(text: str) -> dict:
    res = re.sub(r'```json|```', '', remove_reasoning(text)).strip()
    res = res[res.find('{'):]
    
    # Try to extract the first complete JSON object
    try:
        # Find the position of the first complete JSON by tracking braces
        brace_count = 0
        in_string = False
        escape_next = False
        json_end = -1
        
        for i, char in enumerate(res):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break
        
        if json_end > 0:
            json_str = res[:json_end]
            return json.loads(json_str, strict=False)
        else:
            # Fallback to original behavior
            return json.loads(res, strict=False)
    except json.JSONDecodeError:
        print(text)
        raise

def generate_answer(model: str, token: str, prompt: str,
                    max_tokens: int = None, temperature: float = None,
                    top_p: float = None, reasoning_effort: str = "medium",
                    reasoning_enabled: bool = None) -> tuple[str, str]:
    """Generate an answer from the given model.

    reasoning_enabled:
        - None (default): keep current behavior, send `reasoning_effort`.
        - True: explicitly enable reasoning via `extra_body={"reasoning": {"enabled": True, "effort": ...}}`.
        - False: explicitly disable reasoning via `extra_body={"reasoning": {"enabled": False}}`.
          Hybrid-thinking models on OpenRouter (qwen3, glm-4.5/4.5-air, grok-4.1-fast)
          honor this and skip the chain-of-thought trace.
    """
    global pipe
    reasoning = ''
    generate_answer.last_citations = []

    if model.startswith("hf:"):
        # Huggingface models - token not needed
        hf_model_name = model[3:]  # Remove "hf:" prefix

        try:
            if not pipe or pipe.model.name_or_path != hf_model_name:
                pipe = pipeline(
                    "text-generation",
                    model=hf_model_name,
                    device_map="auto",
                    model_kwargs={
                        "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
                    },
                    trust_remote_code=True
                )

            response = pipe(
                prompt,
                max_new_tokens=max_tokens or 1024,
                do_sample=True,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p if top_p is not None else 0.95,
            )
            raw = response[0]["generated_text"]

            # Extract the part after the prompt
            if raw.startswith(prompt):
                raw = raw[len(prompt):]
        except Exception as e:
            print(f"Huggingface model error: {e}")
            raise
    elif "/" in model and not model.startswith("hf:"):
        # OpenRouter models
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=token,
        )

        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ]},
            ],
            "stream": False,
            "extra_body": {},
        }
        if reasoning_enabled is False:
            # Disable thinking for hybrid-reasoning models (qwen3, glm-4.5-air, grok-4.1-fast).
            create_kwargs["extra_body"]["reasoning"] = {"enabled": False}
        elif reasoning_enabled is True:
            create_kwargs["extra_body"]["reasoning"] = {
                "enabled": True,
                "effort": reasoning_effort,
            }
        else:
            create_kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        if top_p is not None:
            create_kwargs["top_p"] = top_p

        response = client.chat.completions.create(**create_kwargs)

        raw = response.choices[0].message.content or ''
        msg = response.choices[0].message
        reasoning = getattr(msg, 'reasoning', None) or getattr(msg, 'reasoning_content', None) or ''
        if not reasoning and hasattr(msg, 'model_extra') and msg.model_extra:
            reasoning = msg.model_extra.get('reasoning') or msg.model_extra.get('reasoning_content') or ''

        annotations = getattr(msg, 'annotations', None)
        if not annotations and hasattr(msg, 'model_extra') and msg.model_extra:
            annotations = msg.model_extra.get('annotations')
        if annotations:
            generate_answer.last_citations = [
                {
                    "url": a.get("url_citation", {}).get("url", "") if isinstance(a, dict)
                           else getattr(getattr(a, "url_citation", None), "url", ""),
                    "title": a.get("url_citation", {}).get("title", "") if isinstance(a, dict)
                             else getattr(getattr(a, "url_citation", None), "title", ""),
                    "content": a.get("url_citation", {}).get("content", "") if isinstance(a, dict)
                               else getattr(getattr(a, "url_citation", None), "content", ""),
                }
                for a in annotations
            ]
    else:
        # Ollama models
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

def generate_answer_stream(model: str, token: str, prompt: str,
                           max_tokens: int = None, temperature: float = None,
                           top_p: float = None, reasoning_effort: str = "medium",
                           reasoning_enabled: bool = None):
    """Streaming version of generate_answer for OpenRouter models.
    Yields (type, chunk) tuples where type is 'reasoning' or 'content'.

    See `generate_answer` for the meaning of `reasoning_enabled`."""
    if "/" not in model or model.startswith("hf:"):
        answer, reasoning = generate_answer(model, token, prompt,
                                            max_tokens, temperature, top_p, reasoning_effort,
                                            reasoning_enabled=reasoning_enabled)
        if reasoning:
            yield ("reasoning", reasoning)
        yield ("content", answer)
        return

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=token,
    )

    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "extra_body": {},
    }
    if reasoning_enabled is False:
        create_kwargs["extra_body"]["reasoning"] = {"enabled": False}
    elif reasoning_enabled is True:
        create_kwargs["extra_body"]["reasoning"] = {
            "enabled": True,
            "effort": reasoning_effort,
        }
    else:
        create_kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if top_p is not None:
        create_kwargs["top_p"] = top_p

    stream = client.chat.completions.create(**create_kwargs)

    has_native_reasoning = False
    in_think = False
    pending = []
    decided = False
    close_buf = ""

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta

        r = getattr(delta, 'reasoning', None) or getattr(delta, 'reasoning_content', None)
        if not r and hasattr(delta, 'model_extra') and delta.model_extra:
            r = delta.model_extra.get('reasoning') or delta.model_extra.get('reasoning_content')
        if r:
            has_native_reasoning = True
            decided = True
            yield ("reasoning", r)
            continue

        text = delta.content or ''
        if not text:
            continue

        if has_native_reasoning or (decided and not in_think):
            yield ("content", text)
            continue

        if decided and in_think:
            close_buf += text
            if "</think>" in close_buf:
                before, after = close_buf.split("</think>", 1)
                if before:
                    yield ("reasoning", before)
                in_think = False
                close_buf = ""
                if after.strip():
                    yield ("content", after)
            elif len(close_buf) >= 8:
                idx = close_buf.rfind("<")
                if idx >= 0 and "</think>".startswith(close_buf[idx:]):
                    safe = close_buf[:idx]
                    if safe:
                        yield ("reasoning", safe)
                    close_buf = close_buf[idx:]
                else:
                    yield ("reasoning", close_buf)
                    close_buf = ""
            continue

        pending.append(text)
        joined = ''.join(pending)
        stripped = joined.lstrip()

        if stripped.startswith("<think>"):
            decided = True
            in_think = True
            after = stripped.split("<think>", 1)[1]
            pending = []
            if "</think>" in after:
                before, rest = after.split("</think>", 1)
                yield ("reasoning", before)
                in_think = False
                if rest.strip():
                    yield ("content", rest)
            elif after:
                close_buf = after
        elif len(stripped) < 7 and "<think>".startswith(stripped):
            pass
        else:
            decided = True
            yield ("content", joined)
            pending = []

    if close_buf:
        if in_think:
            yield ("reasoning", close_buf)
        elif close_buf.strip():
            yield ("content", close_buf)
    if pending:
        joined = ''.join(pending)
        if joined.strip():
            yield ("content", joined)


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
    # Check if any of the args is a HuggingFace model (second arg is usually the model)
    is_hf_model = any(str(arg).startswith("hf:") for arg in args)
    
    if is_hf_model:
        # For HuggingFace models, don't use token retry mechanism
        print('=' * 20)
        in_data = {"ID":-1,"retries": 0}
        try:
            func(*args, None, **kwargs, in_data=in_data)
            print('=' * 20)
            return True
        except Exception as e:
            print("Unexpected error:")
            traceback.print_exception(e)
            print('=' * 20)
            return False
    
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
    print(tokens)
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
                print("Rate limit exhausted")
                tokens.remove(token)
                random.shuffle(tokens)
                token_cycle = itertools.cycle(tokens)
                print(f"Token \"{token}\" will not be used any more.")
            if e.body["message"].startswith("Rate limit exceeded: free-models-per-min"):
                print("Rate limit exhausted")
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

def run_without_retry(func, *args, **kwargs):
    """Run a function without retry mechanism (for local models like HuggingFace)"""
    print('=' * 100)
    try:
        func(*args, None, **kwargs)  # Pass None as token since it won't be used
        print('=' * 100)
        return True
    except Exception as e:
        print("Unexpected error:")
        traceback.print_exception(e)
        print('=' * 100)
        return False
