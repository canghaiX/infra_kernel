# 7. 练习项目

这一章是最适合你的练习项目清单。前 1-6 章偏“学概念”，这一章偏“动手形成工程直觉”。

## 项目 1：PyTorch 实现 RMSNorm、Softmax、RoPE

目标：

- 用最直接的 PyTorch 写出正确实现。
- 每个函数都做 correctness test。
- 重点看 shape、广播、数值稳定性。

运行：

```bash
python "7. 练习项目/pytorch_ops_practice.py"
```

## 项目 2：Triton 实现 RMSNorm

目标：

- 做 correctness test。
- 跑不同 `hidden_size` benchmark。
- 和 torch 版本对比 latency。

运行：

```bash
python "7. 练习项目/triton_rmsnorm_benchmark.py"
```

## 项目 3：Triton 实现 Softmax

目标：

- 跑不同 `seq_len` benchmark。
- 分析 latency 随序列长度变化的趋势。
- 理解 row-wise softmax 里的 max reduction 和 sum reduction。
- 注意：手写 Triton 不一定总比 torch 快，短序列上 kernel launch overhead 和 torch 内置优化可能更占优势。

运行：

```bash
python "7. 练习项目/triton_softmax_benchmark.py"
```

## 项目 4：CUDA 实现简化版 vector add、reduction、softmax

目标：

- 先写最小可懂的 CUDA kernel。
- 熟悉 `threadIdx`、`blockIdx`、shared memory、block reduction。
- 不追求极限优化，先保证正确。

运行：

```bash
make -C "7. 练习项目" run-cuda
```

## 项目 5：vLLM benchmark

目标：

- 改 `batch_size` 看吞吐变化。
- 改 `seq_len` 看 TTFT 变化。
- 记录 `dtype`，对比 fp16/bf16/fp8 等服务配置。
- 观察 `TTFT`、`decode latency`、`tokens/s`。

这个脚本对 OpenAI-compatible HTTP 接口发请求，因此也能测 vLLM 以外的兼容服务。

先启动一个 vLLM OpenAI-compatible server，然后运行：

```bash
python "7. 练习项目/vllm_benchmark.py" \
  --url http://127.0.0.1:8000/v1/completions \
  --model your-model-name \
  --batch-size 4 \
  --seq-len 512 \
  --max-tokens 64 \
  --dtype bf16
```

如果服务没有启动，脚本会给出提示并退出。
