#!/usr/bin/env python3
"""Compare vLLM decode latency/throughput without counting prefill as decode."""

import argparse
import concurrent.futures
import json
import statistics
import threading
import time
import urllib.request

PROMPTS = {
    "code": (
        "Write a complete Python implementation of a weighted LRU cache. "
        "Include type hints, documentation, and unittest tests for eviction, "
        "oversized entries, and replacement. Output only the code."
    ),
    "prose": (
        "Write a 700-word magazine article explaining how astronomers measure "
        "distances to stars and galaxies. Use concrete examples and cover "
        "parallax, Cepheid variables, and redshift."
    ),
}


def metrics(base, headers=None):
    result = {}
    req = urllib.request.Request(base + "/metrics", headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as response:
        for raw_line in response:
            line = raw_line.decode()
            if not line.startswith("vllm:spec_decode_num_"):
                continue
            labels, value = line.rsplit(None, 1)
            if labels.split("{", 1)[0].endswith("_total"):
                result[labels] = float(value)
    return result


def total(delta, name):
    return sum(
        value for key, value in delta.items()
        if key.split("{", 1)[0] == f"vllm:spec_decode_num_{name}_total"
    )


def request(args, prompt, tokens, barrier):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        args.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **args.headers},
    )
    barrier.wait(timeout=15)
    start = time.monotonic()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    last = time.monotonic()
                    if first is None:
                        first = last
    if first is None or usage is None:
        raise RuntimeError("Expected streamed content and final usage")
    return {
        "first": first,
        "last": last,
        "ttft_s": first - start,
        "tokens": usage["completion_tokens"],
        "prompt_tokens": usage["prompt_tokens"],
    }


def wave(args, prompts, tokens):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        barrier = threading.Barrier(len(prompts))
        jobs = [executor.submit(request, args, prompt, tokens, barrier) for prompt in prompts]
        return [job.result() for job in jobs]


def run(args, workload, concurrency):
    prompts = [
        f"{PROMPTS[workload]}\nThis is benchmark variant {rank + 1}. "
        "Do not mention the variant in the answer."
        for rank in range(concurrency)
    ]
    # Warm the same prompts on every configuration before measuring decode.
    wave(args, prompts, 8)
    for rep in range(args.repeats):
        before = metrics(args.url, args.headers)
        streams = wave(args, prompts, args.tokens)
        after = metrics(args.url, args.headers)
        delta = {key: value - before.get(key, 0.0) for key, value in after.items()}
        drafts = total(delta, "drafts")
        proposed = total(delta, "draft_tokens")
        accepted = total(delta, "accepted_tokens")
        first = min(row["first"] for row in streams)
        last = max(row["last"] for row in streams)
        for row in streams:
            elapsed = row.pop("last") - row.pop("first")
            row["decode_tok_s"] = round((row["tokens"] - 1) / elapsed, 2) if elapsed else None
            row["ttft_s"] = round(row["ttft_s"], 3)
        rates = [row["decode_tok_s"] for row in streams if row["decode_tok_s"] is not None]
        print(json.dumps({
            "workload": workload,
            "concurrency": concurrency,
            "repeat": rep + 1,
            "temperature": args.temperature,
            "median_stream_decode_tok_s": round(statistics.median(rates), 2) if rates else None,
            "aggregate_decode_tok_s": (
                round(sum(row["tokens"] - 1 for row in streams) / (last - first), 2)
                if last > first else None
            ),
            "accepted_per_draft": round(accepted / drafts, 3) if drafts else None,
            "acceptance_pct": round(100 * accepted / proposed, 1) if proposed else None,
            "acceptance_by_position_pct": {
                key.split('position="', 1)[1].split('"', 1)[0]: round(100 * value / drafts, 1)
                for key, value in delta.items()
                if drafts and key.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")
            },
            "streams": streams,
        }), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="baidu/ERNIE-5.5-Flash-Preview")
    parser.add_argument("--workload", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--concurrency", nargs="+", type=int, choices=(1, 2, 3), default=[1, 3])
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--api-key", default="", help="Bearer token for --api-key servers")
    args = parser.parse_args()
    args.headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    args.url = args.url.rstrip("/")
    for concurrency in args.concurrency:
        for workload in args.workload:
            run(args, workload, concurrency)


if __name__ == "__main__":
    main()
