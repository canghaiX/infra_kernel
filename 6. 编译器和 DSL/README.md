# 6. 编译器 / DSL

这一章属于进阶，不建议一开始就啃。更好的顺序是：

1. 先会用 PyTorch 写出正确计算。
2. 再会用 Triton/CUDA 写 kernel。
3. 最后再看 IR、lowering、PTX、编译器优化。

## 这些名词是什么

- `MLIR`：Multi-Level Intermediate Representation，适合表达多层级 IR 和 lowering pipeline。很多编译器项目会用 MLIR 组织从高层算子到低层硬件代码的转换。
- `LLVM`：通用编译器基础设施，LLVM IR 是很多语言/DSL 继续 lowering 的中间层。
- `Triton IR`：Triton kernel 编译后的一种中间表示，比 Python 源码低层，但还没到最终 PTX/SASS。
- `TileLang`：面向 tile 级 GPU kernel 编程的 DSL，目标是更直接表达矩阵 tile、shared memory、pipeline 等模式。
- `PTX`：NVIDIA GPU 的虚拟指令集，可以理解成 CUDA/Triton 编译到 NVIDIA GPU 前的重要低层表示。PTX 之后还会由驱动/JIT 编到更底层的 SASS。

## 典型编译链路

Triton 大致可以理解成：

```text
Python @triton.jit kernel
  -> Triton AST / TTIR
  -> Triton GPU IR / LLVM IR
  -> PTX
  -> SASS
  -> GPU 执行
```

CUDA C++ 大致可以理解成：

```text
CUDA C++
  -> NVVM / LLVM IR
  -> PTX
  -> SASS
  -> GPU 执行
```

## 运行

```bash
python "6. 编译器和 DSL/triton_ir_ptx_intro.py"
```

脚本会编译一个很小的 Triton vector add kernel，然后打印可用的 IR/PTX 片段。不同 Triton 版本暴露的字段可能不同，所以脚本会自动列出当前版本能拿到哪些编译产物。

## 学习建议

1. 不要先背 MLIR/LLVM 的所有概念，先知道它们都服务于 lowering 和优化。
2. 看 IR 时先找 load/store、算术操作、program id、mask。
3. 看 PTX 时先找 `ld`、`st`、`add`、predicate，不急着看懂所有寄存器分配。
4. TileLang、MLIR pass、LLVM pass 都可以后面再深入；入门阶段先建立“高层 DSL 如何变成 GPU 指令”的路径感。
