#!/usr/bin/env python3
"""Verify long-context retrieval and report measured prompt tokens and TTFT."""

import argparse
import json
import math
import random
import time
import urllib.request

FILLER = (
    "north south east west river stone cloud light field harbor silver amber "
    "cedar grass slope tunnel bridge summit dune valley ocean pine granite"
).split()
NEEDLE = "OPAL-4917"


def post(base, path, body, timeout, headers=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def payload(model, words, seed):
    rng = random.Random(seed)
    filler = rng.choices(FILLER, k=words)
    midpoint = words // 2
    prompt = (
        " ".join(filler[:midpoint])
        + f"\n\nThe inventory verification code is {NEEDLE}.\n\n"
        + " ".join(filler[midpoint:])
        + "\n\nWhat is the inventory verification code? Reply with the code only."
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }


def prompt_for_size(args):
    probe_words = 400
    with post(args.url, "/tokenize", payload(args.model, probe_words, args.seed), 60, args.headers) as r:
        probe_tokens = json.load(r)["count"]
    words = math.ceil(args.tokens * probe_words / probe_tokens)
    for _ in range(5):
        body = payload(args.model, words, args.seed)
        with post(args.url, "/tokenize", body, 120, args.headers) as r:
            prompt_tokens = json.load(r)["count"]
        if args.tokens - max(64, args.tokens // 200) <= prompt_tokens <= args.tokens:
            return body, prompt_tokens
        error = args.tokens - min(32, args.tokens // 1000) - prompt_tokens
        words += int(error * words / prompt_tokens)
    raise RuntimeError(f"Could not size the prompt (last estimate: {prompt_tokens})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="baidu/ERNIE-5.5-Flash-Preview")
    parser.add_argument("--tokens", type=int, default=390000)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--api-key", default="", help="Bearer token for --api-key servers")
    args = parser.parse_args()
    args.headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    args.url = args.url.rstrip("/")
    if not 1000 <= args.tokens <= 399960:
        parser.error("--tokens must be between 1000 and 399960")

    body, estimated = prompt_for_size(args)
    body.update({
        "temperature": 0, "max_tokens": 24, "stream": True,
        "stream_options": {"include_usage": True},
    })
    print(f"Submitting {estimated:,} prompt tokens", flush=True)
    start = time.monotonic()
    first = None
    usage = None
    text = []
    with post(args.url, "/v1/chat/completions", body, 1800, args.headers) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            event = line[5:].strip()
            if event == b"[DONE]":
                break
            chunk = json.loads(event)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first is None:
                        first = time.monotonic()
                    text.append(content)
    answer = "".join(text).strip()
    result = {
        "prompt_tokens": usage and usage["prompt_tokens"],
        "ttft_s": round(first - start, 2) if first is not None else None,
        "elapsed_s": round(time.monotonic() - start, 2),
        "answer": answer,
        "passed": answer == NEEDLE,
    }
    print(json.dumps(result), flush=True)
    if not result["passed"] or not usage or usage["prompt_tokens"] < estimated - 32:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
