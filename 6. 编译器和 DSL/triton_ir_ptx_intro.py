"""
编译器 / DSL 入门：观察 Triton IR、LLVM IR 和 PTX。

这份脚本做一件事：
1. 定义一个最小 Triton vector add kernel。
2. 运行一次触发 JIT 编译。
3. 从 compiled kernel 中尽量取出当前 Triton 版本暴露的 IR/PTX。

不同 Triton 版本的内部字段可能不同，所以这份脚本写得比较防御式：
- 能拿到什么就打印什么。
- 拿不到时告诉你当前版本暴露了哪些字段。
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.runtime import driver


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """一个极小 kernel，方便观察编译结果。"""

    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def print_section(title: str, text: str, max_lines: int = 80) -> None:
    """打印 IR/PTX 片段，避免一次刷太多内容。"""

    print(f"\n=== {title} ===")
    lines = text.splitlines()
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"... 省略 {len(lines) - max_lines} 行 ...")


def try_get_compiled_artifacts(kernel: Any) -> dict[str, str]:
    """
    尝试从 Triton JITFunction 中取编译产物。

    常见位置：
    - kernel.cache
    - compiled_kernel.asm

    这些不是稳定 public API，所以只适合作为学习观察工具。
    """

    artifacts: dict[str, str] = {}

    cache = getattr(kernel, "cache", None)
    if cache is None:
        return artifacts

    # Triton cache 通常是多层 dict：
    # device -> specialization -> compiled kernel
    stack: list[Any] = [cache]
    visited: set[int] = set()

    while stack:
        obj = stack.pop()
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue

        asm = getattr(obj, "asm", None)
        if isinstance(asm, dict):
            for name, value in asm.items():
                if isinstance(value, str):
                    artifacts[str(name)] = value

    return artifacts


def compile_with_ast_source(block_size: int, n_elements: int) -> dict[str, str]:
    """
    Triton 3.x 推荐用 ASTSource 显式编译 JITFunction。

    signature 描述运行时参数类型：
    - *fp32 表示 float32 指针。

    constexprs 描述编译期常量：
    - n_elements 和 BLOCK_SIZE 在这个示例里都作为常量参与编译。
    """

    source = ASTSource(
        vector_add_kernel,
        signature={
            "x_ptr": "*fp32",
            "y_ptr": "*fp32",
            "out_ptr": "*fp32",
        },
        constexprs={
            "n_elements": n_elements,
            "BLOCK_SIZE": block_size,
        },
    )
    compiled = triton.compile(source, target=driver.active.get_current_target())
    asm = getattr(compiled, "asm", {})
    return {str(key): value for key, value in asm.items() if isinstance(value, str)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能编译 Triton kernel。")

    torch.manual_seed(0)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    block_size = 256
    grid = (triton.cdiv(n, block_size),)

    # 第一次调用会触发 Triton JIT 编译。
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    max_error = (out - (x + y)).abs().max().item()
    print("vector_add max_error:", max_error)

    artifacts = compile_with_ast_source(block_size=block_size, n_elements=n)
    if not artifacts:
        artifacts = try_get_compiled_artifacts(vector_add_kernel)
    if not artifacts:
        print("\n没有从当前 Triton 版本中取到 asm/IR 字段。")
        print("可观察字段:", sorted(dir(vector_add_kernel))[:80])
        return

    print("\n当前 Triton 暴露的编译产物:", sorted(artifacts))

    # 常见 key 包括 ttir、ttgir、llir、ptx、cubin。
    preferred_order = ["ttir", "ttgir", "llir", "ptx"]
    printed = set()

    for key in preferred_order:
        if key in artifacts:
            print_section(key, artifacts[key])
            printed.add(key)

    for key, value in artifacts.items():
        if key not in printed and isinstance(value, str):
            print_section(key, value, max_lines=30)

    print("\n阅读提示:")
    print("- TTIR/TTGIR 里先找 load/store、program_id、mask。")
    print("- LLVM IR 里先找 getelementptr、load、store、fadd。")
    print("- PTX 里先找 ld、st、add 和 predicate，例如 @%p。")


if __name__ == "__main__":
    main()
