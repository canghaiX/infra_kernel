# 5. 通信和分布式

做 DeepSeek、MoE、大模型训练/推理优化时，通信和分布式非常重要。单卡算子优化解决的是“每张卡怎么快”，分布式解决的是“多张卡怎么一起快”。

## 先看这些概念

- `NCCL`：NVIDIA GPU 间通信库，PyTorch 多 GPU 通常用它做 collective communication。
- `HCCL`：华为 Ascend 生态里的集合通信库，概念上和 NCCL 类似，但硬件和软件栈不同。
- `AllReduce`：所有 rank 的 tensor 求和/归约，然后每个 rank 都拿到完整结果。
- `AllGather`：每个 rank 拿一片数据，最后每个 rank 都收集到所有片。
- `ReduceScatter`：先 reduce，再把结果按 rank 切分发回去。
- `All-to-All`：每个 rank 给每个 rank 发不同切片，MoE expert dispatch 常见。
- `TP`：Tensor Parallel，把单层大矩阵按列/行切到多卡。
- `PP`：Pipeline Parallel，把不同层切到不同卡。
- `DP`：Data Parallel，每张卡一份模型，处理不同 batch，梯度用 AllReduce 同步。
- `EP`：Expert Parallel，MoE 里不同 expert 放到不同卡，请求按 token 路由。

## 运行

先看不需要多进程的 shape 演示：

```bash
python "5. 通信和分布式/parallelism_concepts.py"
```

再用 `torchrun` 跑 collective 通信。单机 2 卡：

```bash
torchrun --standalone --nproc_per_node=2 "5. 通信和分布式/distributed_collectives_intro.py"
```

如果没有可用 GPU，可以用 CPU/gloo：

```bash
DIST_BACKEND=gloo torchrun --standalone --nproc_per_node=2 "5. 通信和分布式/distributed_collectives_intro.py"
```

## 学习建议

1. 先跑 `AllReduce`，理解“每张卡最后都拿到一样的结果”。
2. 再跑 `AllGather` 和 `ReduceScatter`，它们经常和 Tensor Parallel 绑定。
3. 最后看 `All-to-All`，它是 MoE / Expert Parallel 的核心通信模式。
4. 看 TP/PP/DP/EP 时，不要先背名字，先问：模型、数据、激活、token 分别被切到了哪里？
