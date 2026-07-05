# 2. 大模型推理流程

这一章关注推理系统的核心流程。要理解推理优化，先要分清两个阶段：

- `prefill`：一次性处理用户输入的 prompt，建立第一批 KV cache，并产出第一个 next-token logits。
- `decode`：每轮只处理新生成的 1 个 token，同时复用 KV cache，不再重复计算整段历史。

## 关键概念

- `KV cache`：缓存每层 attention 的 K/V，decode 时只需要计算新 token 的 Q/K/V。
- `batching`：把多个请求放在同一个 batch 中算，提高吞吐，但可能影响单请求延迟。
- `TTFT`：time to first token，用户从发请求到看到第一个 token 的时间。
- `TPOT`：time per output token，生成阶段平均每个 token 的耗时。
- `tokens/s`：吞吐，单位时间生成多少 token。
- 显存占用：主要包括模型权重、KV cache、临时激活；长上下文和大 batch 会显著增加 KV cache。

## 运行

```bash
python "2. 大模型推理流程/llm_inference_flow.py"
```

脚本使用一个极简 Transformer block，不追求模型质量，只用来把推理流程讲清楚。
