# AI Infra 算子层入门练习

这个项目用于补齐 AI Infra / 推理优化入门前需要掌握的几块基础：

1. `PyTorch` 和 tensor 计算
2. 大模型推理流程
3. `Triton` 算子入门
4. `CUDA` 算子入门
5. 通信和分布式
6. 编译器 / DSL
7. 练习项目

推荐先把代码跑起来，再对照注释和输出理解每一步的 shape、计算含义和性能指标。

## 目录

- [1. PyTorch 和 tensor 计算](./1.%20PyTorch%20和%20tensor%20计算)
  - `tensor_ops_intro.py`：演示 `matmul`、`attention`、`softmax`、`rmsnorm`、`rope`、`SiLU/SwiGLU`，重点看 shape 怎么变。
- [2. 大模型推理流程](./2.%20大模型推理流程)
  - `llm_inference_flow.py`：用一个极简 Transformer block 演示 `prefill`、`decode`、`KV cache`、`batching`、`TTFT`、`TPOT`、`tokens/s` 和显存占用估算。
- [3. Triton](./3.%20Triton)
  - `triton_ops_intro.py`：用 Triton 写 `vector add`、`softmax`、`RMSNorm`、`RoPE`、`MatMul`，同时理解 `block`、`tile`、`mask`、`load/store`、`parallel reduction`。
- [4. CUDA](./4.%20CUDA)
  - `cuda_ops_intro.cu`：用 CUDA C++ 写 `vector add`、`matrix transpose`、`reduction`、`softmax`、`rmsnorm`、`matmul` 和 shared memory tiled matmul。
  - `Makefile`：用于编译 CUDA 示例。
- [5. 通信和分布式](./5.%20通信和分布式)
  - `distributed_collectives_intro.py`：用 `torch.distributed` 演示 `AllReduce`、`AllGather`、`ReduceScatter`、`All-to-All`。
  - `parallelism_concepts.py`：不用多进程，直接用 tensor shape 演示 `TP/PP/DP/EP` 的切分直觉。
- [6. 编译器和 DSL](./6.%20编译器和%20DSL)
  - `triton_ir_ptx_intro.py`：编译一个 Triton kernel，打印 `Triton IR`、`LLVM IR` 和 `PTX` 片段。
  - `README.md`：解释 `MLIR`、`LLVM`、`Triton IR`、`TileLang`、`PTX` 分别处在编译链路的什么位置。
- [7. 练习项目](./7.%20练习项目)
  - `pytorch_ops_practice.py`：PyTorch 实现并测试 `RMSNorm`、`Softmax`、`RoPE`。
  - `triton_rmsnorm_benchmark.py`：Triton 实现 RMSNorm，做 correctness test、不同 hidden size benchmark，并和 torch 对比。
  - `triton_softmax_benchmark.py`：Triton 实现 Softmax，做不同 `seq_len` benchmark，并分析 latency。
  - `cuda_simplified_ops.cu`：CUDA 实现简化版 `vector add`、`reduction`、`softmax`。
  - `vllm_benchmark.py`：对 OpenAI-compatible/vLLM 服务做 `batch_size`、`seq_len`、`dtype`、`TTFT`、`decode latency`、`tokens/s` benchmark。

## 环境

前两章只依赖 PyTorch：

```bash
pip install torch
```

Triton 章节需要：

```bash
pip install triton
```

CUDA 章节需要本机有 NVIDIA GPU、CUDA driver 和 `nvcc`。

分布式章节需要 PyTorch distributed。单机多卡可以用 `nccl`，CPU 或无 NCCL 环境可以用 `gloo`。

如果你已经在深度学习环境里，通常可以直接运行：

```bash
python "1. PyTorch 和 tensor 计算/tensor_ops_intro.py"
python "2. 大模型推理流程/llm_inference_flow.py"
python "3. Triton/triton_ops_intro.py"
make -C "4. CUDA" run
python "5. 通信和分布式/parallelism_concepts.py"
torchrun --standalone --nproc_per_node=2 "5. 通信和分布式/distributed_collectives_intro.py"
python "6. 编译器和 DSL/triton_ir_ptx_intro.py"
python "7. 练习项目/pytorch_ops_practice.py"
python "7. 练习项目/triton_rmsnorm_benchmark.py"
python "7. 练习项目/triton_softmax_benchmark.py"
make -C "7. 练习项目" run-cuda
```

## 学习顺序

1. 先看 tensor 的 shape：`[batch, seq_len, hidden]`、`[batch, heads, seq_len, head_dim]` 是大模型里最常见的两套视角。
2. 再看 attention：`Q @ K^T -> softmax -> @ V` 是后续 CUDA kernel 优化、FlashAttention、KV cache 的共同基础。
3. 然后看推理流程：`prefill` 处理整段 prompt，`decode` 每次生成一个 token；推理优化大多围绕减少 decode 阶段的延迟和显存压力。
4. 再学 Triton：它比 CUDA C++ 更接近 Python，适合先建立 block、tile、mask、load/store、parallel reduction 的直觉。
5. 最后学 CUDA：理解 thread/block/grid、shared memory、同步和访存优化后，再看 Triton 或推理框架源码会轻松很多。
6. 如果做多卡推理或训练，再补通信和分布式：collective 通信是 TP/PP/DP/EP 的地基。
7. 编译器 / DSL 放后面：先知道 kernel 怎么写、怎么跑，再去看 IR、lowering、PTX 会更自然。
8. 最后用练习项目串起来：正确性、benchmark、延迟分析和服务端吞吐测试都要做，才会真正进入 AI Infra 的工作方式。
