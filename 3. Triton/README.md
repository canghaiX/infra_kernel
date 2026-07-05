# 3. Triton

Triton 比 CUDA C++ 更适合大模型算子入门：它仍然需要你思考并行、访存和 tile，但语法更接近 Python，调试成本低很多。

这一章先写这些算子：

- `RMSNorm`
- `Softmax`
- `RoPE`
- `MatMul`

同时理解这些概念：

- `block`：一个 Triton program 处理的一块数据。
- `tile`：矩阵乘法等算子中，每次搬运和计算的小矩阵块。
- `mask`：处理越界、非 2 的幂长度、非整除 tile 时的保护条件。
- `memory load/store`：从显存读写数据，优化重点通常是少读、连续读、合并读。
- `parallel reduction`：并行求和、求最大值，是 softmax、norm、reduction 的基础。

## 运行

```bash
python "3. Triton/triton_ops_intro.py"
```

脚本会把 Triton kernel 的输出和 PyTorch reference 对比，输出最大误差。

## 阅读顺序

1. 先看 `vector_add_kernel`，理解 `program_id`、`offsets`、`mask`、`tl.load`、`tl.store`。
2. 再看 `softmax_kernel`，理解一行数据内的 max/sum reduction。
3. 然后看 `rmsnorm_kernel`，它和大模型里的 RMSNorm 直接对应。
4. 再看 `rope_kernel`，理解每个 token/head 内的成对旋转。
5. 最后看 `matmul_kernel`，重点看 `BLOCK_M/BLOCK_N/BLOCK_K` 和 `tl.dot`。
