# 1. PyTorch 和 tensor 计算

这一章要先看懂大模型算子背后的 tensor 运算。之后写 CUDA 或看推理框架时，很多优化其实都是在优化这些基础操作的访存、并行度和 kernel 融合。

## 你需要掌握的点

- `matmul`：矩阵乘法，是线性层、QKV 投影、attention 分数计算的基础。
- `attention`：核心公式是 `softmax(QK^T / sqrt(head_dim))V`。
- `softmax`：把 logits 变成概率分布，常用于 attention 权重和采样。
- `rmsnorm`：大模型常用归一化，比 LayerNorm 少了减均值步骤。
- `rope`：旋转位置编码，把位置信息注入 Q/K。
- `silu/swiglu`：常见 MLP 激活与门控结构。
- shape 变化：重点追踪 `[batch, seq_len, hidden]` 到 `[batch, heads, seq_len, head_dim]` 的变化。

## 运行

```bash
python "1. PyTorch 和 tensor 计算/tensor_ops_intro.py"
```

建议运行后逐段看输出，每个函数都打印了关键 shape。
