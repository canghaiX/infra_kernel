"""
练习项目 5：vLLM / OpenAI-compatible HTTP benchmark。

这个脚本不直接 import vLLM，而是请求 OpenAI-compatible endpoint。
好处是：
- vLLM、SGLang、TGI 或自研兼容服务都能测。
- benchmark 脚本不受服务端启动参数变化影响。

示例：
    python "7. 练习项目/vllm_benchmark.py" \
      --url http://127.0.0.1:8000/v1/completions \
      --model your-model-name \
      --batch-size 4 \
      --seq-len 512 \
      --max-tokens 64 \
      --dtype bf16
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import aiohttp


@dataclass
class RequestMetrics:
    """单个请求的指标。"""

    ttft_s: float
    total_s: float
    output_tokens: int

    @property
    def decode_latency_s(self) -> float:
        """首 token 后到请求结束的时间。"""

        return max(self.total_s - self.ttft_s, 0.0)

    @property
    def tokens_per_s(self) -> float:
        """单请求输出吞吐。"""

        if self.total_s == 0:
            return float("inf")
        return self.output_tokens / self.total_s


def build_prompt(seq_len: int) -> str:
    """
    构造近似长度的 prompt。

    这里的 seq_len 是练习用近似值，不等于 tokenizer 精确 token 数。
    真正严谨的 benchmark 应该使用目标模型 tokenizer 生成固定 token 数。
    """

    words = ["benchmark"] * seq_len
    return " ".join(words)


async def run_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestMetrics:
    """
    发起一个 streaming completion 请求。

    TTFT:
    - 从发出请求到收到第一个非空 token chunk 的时间。

    decode latency:
    - 从第一个 token 到流式响应结束的时间。
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.perf_counter()
    first_token_time: float | None = None
    output_tokens = 0

    async with session.post(url, json=payload) as response:
        if response.status != 200:
            text = await response.text()
            raise RuntimeError(f"HTTP {response.status}: {text[:500]}")

        async for raw_line in response.content:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            # OpenAI-compatible completions streaming:
            # choices[0].text 通常是当前 chunk 文本。
            choices = event.get("choices") or []
            if choices:
                text = choices[0].get("text") or ""
                if text and first_token_time is None:
                    first_token_time = time.perf_counter()

            # 部分服务会在最后一个 chunk 返回 usage。
            usage = event.get("usage") or {}
            completion_tokens = usage.get("completion_tokens")
            if isinstance(completion_tokens, int):
                output_tokens = completion_tokens

    total_s = time.perf_counter() - start
    ttft_s = (first_token_time - start) if first_token_time is not None else total_s

    # 如果服务端不返回 usage，就用 max_tokens 做近似。
    if output_tokens == 0:
        output_tokens = max_tokens

    return RequestMetrics(ttft_s=ttft_s, total_s=total_s, output_tokens=output_tokens)


async def run_benchmark(args: argparse.Namespace) -> list[RequestMetrics]:
    """并发发起 batch_size 个请求，模拟一个 batch 的压力。"""

    prompt = build_prompt(args.seq_len)
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            run_one_request(
                session=session,
                url=args.url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for _ in range(args.batch_size)
        ]
        return await asyncio.gather(*tasks)


def summarize(args: argparse.Namespace, metrics: list[RequestMetrics], wall_s: float) -> None:
    """打印 benchmark 汇总。"""

    ttft_ms = [m.ttft_s * 1000 for m in metrics]
    decode_ms = [m.decode_latency_s * 1000 for m in metrics]
    total_tokens = sum(m.output_tokens for m in metrics)
    aggregate_tokens_per_s = total_tokens / wall_s if wall_s > 0 else float("inf")

    print("\n=== vLLM / OpenAI-compatible benchmark ===")
    print(f"url: {args.url}")
    print(f"model: {args.model}")
    print(f"batch_size: {args.batch_size}")
    print(f"seq_len approx: {args.seq_len}")
    print(f"dtype label: {args.dtype}")
    print(f"max_tokens: {args.max_tokens}")

    print("\n=== latency ===")
    print(f"TTFT avg: {statistics.mean(ttft_ms):.2f} ms")
    print(f"TTFT p50: {statistics.median(ttft_ms):.2f} ms")
    print(f"TTFT max: {max(ttft_ms):.2f} ms")
    print(f"decode latency avg: {statistics.mean(decode_ms):.2f} ms")
    print(f"request wall time: {wall_s * 1000:.2f} ms")

    print("\n=== throughput ===")
    print(f"total output tokens: {total_tokens}")
    print(f"aggregate tokens/s: {aggregate_tokens_per_s:.2f}")
    print("说明: 如果服务端没有返回 usage，output tokens 会按 max_tokens 近似。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vLLM/OpenAI-compatible completions endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="unknown", help="记录服务端 dtype 配置，例如 fp16/bf16/fp8。")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        start = time.perf_counter()
        metrics = asyncio.run(run_benchmark(args))
        wall_s = time.perf_counter() - start
    except aiohttp.ClientConnectorError as exc:
        print(f"连接失败: {exc}")
        print("请先启动 vLLM/OpenAI-compatible server，再运行本脚本。")
        return
    except RuntimeError as exc:
        print(f"请求失败: {exc}")
        return

    summarize(args, metrics, wall_s)


if __name__ == "__main__":
    main()
