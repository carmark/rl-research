# verl Agentic RL 训练 — Infra 准备方案

> 日期: 2026-04-26 | 目标: 基于 verl 训练工具调用 Agent | 基座: DeepSeek MoE | 集群: 64+ GPU

---

## 一、硬件层

### GPU 集群

| 阶段 | GPU | 数量 | 用途 |
|------|-----|------|------|
| **验证阶段** | A100/H800 80GB | 8 (1节点) | Qwen2.5-7B + FSDP 跑通全流程 |
| **扩展阶段** | H800 80GB | 64 (8节点) | DeepSeek-V2.5 (236B MoE) 正式训练 |
| **生产阶段** | H800 80GB | 128+ (16节点) | DeepSeek-V3 (671B) / 解耦模式 |

关键选型要点：
- **必须 80GB 显存** — DeepSeek MoE 即使用 TP=8，单卡显存也很紧张（236B 模型 FP8 推理约占 30GB/卡）
- **H800 > A100** — H800 的 NVLink 带宽 (900GB/s) 对 TP 通信更友好；FP8 计算性能也更强
- 如果用 A100，671B 模型必须启用 LoRA

### 网络

| 组件 | 要求 | 理由 |
|------|------|------|
| **节点内** | NVLink/NVSwitch | TP 通信 (每步都要 AllReduce)，延迟敏感 |
| **节点间** | InfiniBand 200/400Gbps **或** RoCE v2 | PP 流水线通信 + 权重同步 (3D-HybridEngine 需要 <300ms) |
| **RDMA** | 必须支持 GPUDirect RDMA | NCCL/NIXL 依赖，否则权重同步会成为严重瓶颈 |

这是最容易被低估的部分。如果节点间只有以太网 (25/100Gbps)，权重同步延迟会从 ~300ms 恶化到秒级，整体吞吐量可能下降 40-60%。

### 存储

| 用途 | 类型 | 容量 | 说明 |
|------|------|------|------|
| 模型权重 | 共享 NFS/Lustre | 2-5TB | DeepSeek-V3 权重 ~1.3TB (FP8)，需要所有节点可访问 |
| Checkpoint | 高速并行文件系统 | 5-10TB | 每个 checkpoint ~500GB (236B), 保留 5 个 |
| 训练数据 | NFS | 100GB+ | Prompt 数据集 + 工具 corpus |
| 日志/监控 | 本地 SSD | 500GB/节点 | WandB 日志、训练 metrics |

### CPU & 内存

| 组件 | 配置 | 理由 |
|------|------|------|
| CPU | 64+ 核/节点 | Ray Worker 调度 + 数据预处理 + 工具沙箱执行 |
| 内存 | 512GB+/节点 | Megatron 权重加载 + KV Cache offload + 数据缓冲区 |

---

## 二、软件栈

### 核心依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                    你的训练代码                          │
│            (verl-agent-training/src/)                    │
├─────────────────────────────────────────────────────────┤
│  verl v0.7.1+          │  你自定义的 AgentLoop/Reward   │
├─────────────────────────────────────────────────────────┤
│  推理引擎              │  训练引擎       │  编排        │
│  vLLM ≥ 0.6            │  Megatron-LM    │  Ray ≥ 2.9   │
│  (或 SGLang)           │  (或 FSDP2)     │              │
├─────────────────────────────────────────────────────────┤
│  PyTorch ≥ 2.3         │  NCCL ≥ 2.20    │  CUDA ≥ 12.1 │
├─────────────────────────────────────────────────────────┤
│  NVIDIA Driver ≥ 535   │  cuDNN ≥ 8.9    │  InfiniBand   │
└─────────────────────────────────────────────────────────┘
```

### 各组件版本与安装

**1) 基础环境**

```bash
# NVIDIA Driver (所有节点)
nvidia-smi  # 确认 Driver ≥ 535, CUDA ≥ 12.1

# PyTorch (推荐 conda 环境)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Flash Attention 2 (MoE 推理/训练加速)
pip install flash-attn --no-build-isolation
```

**2) verl + 推理/训练引擎**

```bash
# verl (核心框架)
pip install "verl>=0.7.1"

# vLLM (推理引擎 — DeepSeek MoE 支持)
pip install vllm

# Megatron-LM (训练引擎 — 5D 并行)
pip install megatron-core
# 或从源码安装以获取最新 MoE/EP 支持:
# git clone https://github.com/NVIDIA/Megatron-LM && cd Megatron-LM && pip install -e .
```

**3) 分布式编排**

```bash
# Ray (集群管理)
pip install "ray[default]>=2.9"

# NCCL (通常随 PyTorch 安装, 但集群需确认版本一致)
# 检查: python -c "import torch; print(torch.cuda.nccl.version())"
```

**4) 工具沙箱**

```bash
# Docker (代码执行隔离)
# 所有节点需安装 Docker Engine
docker build -t verl-sandbox:latest -f docker/Dockerfile.sandbox .
```

**5) 监控**

```bash
pip install wandb  # 训练曲线可视化
```

---

## 三、集群编排

### Ray 集群拓扑

```
┌──────────────────── Head Node ────────────────────┐
│  Ray Head  (GCS + Dashboard :8265)                │
│  RLTrainer (单控制器)                              │
│  8× GPU — 参与训练+推理                           │
├───────────────────────────────────────────────────┤

┌─── Worker Node 1 ───┐  ┌─── Worker Node 2 ───┐  ...  ┌─── Worker Node 7 ───┐
│  Ray Worker          │  │  Ray Worker          │       │  Ray Worker          │
│  8× GPU              │  │  8× GPU              │       │  8× GPU              │
└──────────────────────┘  └──────────────────────┘       └──────────────────────┘
```

启动步骤：

```bash
# Head 节点
ray start --head --num-gpus 8 --dashboard-host 0.0.0.0

# 每个 Worker 节点
ray start --address <HEAD_IP>:6379 --num-gpus 8
```

### GPU 分配策略 (共置模式, 64 GPU)

verl 的 `ResourcePool` + `placement group` 会自动管理，但你需要理解分配逻辑：

```
64 GPU 全部参与训练和推理 (时分复用):

训练阶段: Megatron 5D 并行
  TP=4 (4 GPU 一组做张量并行)
  PP=2 (2 组串联做流水线并行)
  DP=8 (8 份数据并行)
  → 4 × 2 × 8 = 64 GPU ✓

推理阶段: vLLM Server
  TP=8 (8 GPU 一组做张量并行)
  → 8 个 vLLM 实例, 每实例 8 GPU
  → 3D-HybridEngine 自动将 PP 维度转为 micro-DP
```

---

## 四、DeepSeek MoE 特殊准备

这是最需要注意的部分，MoE 模型比 Dense 模型的 infra 要求高很多：

### 4.1 模型权重下载

```bash
# DeepSeek-V2.5 (236B MoE, ~470GB FP16 / ~235GB FP8)
huggingface-cli download deepseek-ai/DeepSeek-V2.5 --local-dir /shared/models/deepseek-v2.5

# 确保所有节点都能通过共享文件系统访问
```

### 4.2 Expert Parallelism (EP) 配置

MoE 模型需要 EP 来分散专家参数。verl + Megatron 的 EP 要求：
- EP 大小通常 = 节点内 GPU 数 (8) 或跨节点 (取决于网络)
- EP 通信走 All-to-All，对节点间带宽敏感
- **需要安装 DeepEP**（如果使用 FP8 MoE 推理加速）

### 4.3 路由一致性监控

训练和推理使用不同引擎 (Megatron vs vLLM)，MoE Router 可能选择不同专家。需要：
- 在训练配置中开启 `moe.monitor_routing_consistency: true`
- 如差异率 > 5%，考虑实现 R3 路由回放 (参考 MILES)

---

## 五、工具调用环境准备

### 5.1 沙箱环境

```bash
# 构建沙箱 Docker 镜像 (所有训练节点)
docker build -t verl-sandbox:latest -f docker/Dockerfile.sandbox .

# 验证
docker run --rm --network=none verl-sandbox:latest python3 -c "print('sandbox ok')"
```

### 5.2 工具后端

| 工具 | 开发阶段 | 生产阶段 |
|------|---------|---------|
| web_search | mock corpus | SerpAPI / Bing API (需 API Key) |
| code_executor | subprocess | Docker 容器 (每次调用独立容器) |
| calculator | 本地 eval | 同 |
| database | SQLite in-memory | 预加载任务数据库 |

### 5.3 LLM-as-Judge (可选)

如果启用 LLM-as-Judge 奖励：
- 需要一个可用的 OpenAI-compatible API 端点
- 或部署一个独立的 Judge LLM (如 GPT-4 / Claude API)
- 预算：每 1000 条轨迹评估约 $5-15（GPT-4 价格）

---

## 六、分阶段落地路径

### 阶段 1 — 单机验证 (1-2 周)

```
硬件: 1 节点, 8× A100/H800
模型: Qwen2.5-7B-Instruct (Dense, 小模型)
训练: FSDP 后端 (无需 Megatron)
目标: 跑通 AgentLoop + 工具调用 + 奖励函数 + GRPO 训练
```

这个阶段只需一台 8-GPU 服务器，验证全流程正确性。

### 阶段 2 — 小规模 MoE (2-3 周)

```
硬件: 2 节点, 16× H800
模型: DeepSeek-V2-Lite 或 Qwen2.5-MoE
训练: Megatron 后端, TP=8, PP=1, DP=2
目标: 验证 MoE EP 并行 + 权重同步
```

### 阶段 3 — 全规模训练 (持续)

```
硬件: 8 节点, 64× H800
模型: DeepSeek-V2.5 (236B MoE)
训练: Megatron, TP=4, PP=2, DP=8, EP=8
目标: 正式 Agentic RL 训练
```

---

## 七、清单总结

| 类别 | 必须 | 推荐 | 可选 |
|------|------|------|------|
| **GPU** | 8× 80GB (验证) | 64× H800 (训练) | 128+ (生产) |
| **网络** | 节点内 NVLink | 节点间 IB 200Gbps | IB 400Gbps + GPUDirect RDMA |
| **存储** | NFS 2TB (模型) | Lustre 10TB (Checkpoint) | 高速 SSD Tier |
| **软件** | verl, vLLM, PyTorch, Ray | Megatron-LM, NCCL, WandB | DeepEP, NIXL |
| **沙箱** | subprocess (开发) | Docker (训练) | Kubernetes (生产) |
| **监控** | WandB | Prometheus + Grafana | 自定义 MoE 路由监控 |

最关键的瓶颈通常不是 GPU 数量，而是**节点间网络带宽**和**共享存储性能**。如果 IB 网络和 Lustre/GPFS 存储到位，64 GPU 的训练集群搭建本身并不复杂，verl + Ray 会处理大部分编排逻辑。
