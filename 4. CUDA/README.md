# 4. CUDA

Triton 入门后再看 CUDA，会更容易理解 CUDA C++ 的细节。CUDA 更底层，你需要直接管理：

- `threadIdx` / `blockIdx`
- `gridDim` / `blockDim`
- global memory 和 shared memory
- `__syncthreads()`
- block 内 reduction
- coalesced memory access

这一章先写：

- `vector add`
- `matrix transpose`
- `reduction`
- `softmax`
- `layernorm/rmsnorm` 里的 `rmsnorm`
- `matmul` 基础版
- `shared memory` tiled matmul

## 运行

```bash
make -C "4. CUDA" run
```

也可以手动编译：

```bash
nvcc -O2 -arch=native "4. CUDA/cuda_ops_intro.cu" -o "4. CUDA/cuda_ops_intro"
"4. CUDA/cuda_ops_intro"
```

如果你的 `nvcc` 不支持 `-arch=native`，可以改成具体架构，例如 `-arch=sm_86`。

## 阅读顺序

1. `vector_add_kernel`：理解 thread/block/grid 的一维映射。
2. `transpose_kernel`：理解二维 block 和 shared memory。
3. `reduce_sum_kernel`：理解 parallel reduction。
4. `row_softmax_kernel`：理解 max reduction + sum reduction。
5. `rmsnorm_kernel`：理解归一化类算子的结构。
6. `matmul_naive_kernel`：理解矩阵乘法最直接的写法。
7. `matmul_tiled_kernel`：理解 shared memory 如何减少 global memory 访问。
