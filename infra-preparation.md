# verl Agentic RL 训练 — Infra 准备方案

> 日期: 2026-04-26 | 目标: 基于 verl 训练工具调用 Agent | 基座: DeepSeek MoE | 集群: 64+ GPU
>
> 相关文档: [verl-integration.md](verl-agent-training/verl-integration.md) (AgentLoop/Reward/Tool 接口详解)

---

## 零、Infra 全景：AgentLoop 视角

Agentic RL 的 infra 与传统 LLM RL 最大的区别在于 **AgentLoop**——Rollout 不再是一次性生成，而是多轮 LLM 推理 + 工具执行的交替循环。这对 infra 的每一层都有额外要求：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PPOTrainer (训练主循环)                          │
│                                                                         │
│  ┌── AgentLoopManager ────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  AgentLoopWorker ×N (Ray Actor, CPU 进程)                          │ │
│  │    │                                                                │ │
│  │    ├─ 协程 Pool (每个 sample 一个 async 协程)                       │ │
│  │    │    │                                                           │ │
│  │    │    ├─ AsyncLLMServerManager.generate()                        │ │
│  │    │    │    └─ HTTP → vLLM/SGLang Server (GPU)                    │ │
│  │    │    │         └─ Sticky Session (同一 request_id → 同一 Server) │ │
│  │    │    │                                                           │ │
│  │    │    ├─ ToolParser.parse() → BaseTool.execute()                 │ │
│  │    │    │    └─ subprocess / Docker / 外部 API (CPU)                │ │
│  │    │    │                                                           │ │
│  │    │    └─ 拼接 tool response → 下一轮 generate()                  │ │
│  │    │                                                                │ │
│  │    └─ 收集 AgentLoopOutput (prompt_ids, response_ids, mask, ...)   │ │
│  │                                                                     │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  Reward (compute_score) → Advantage (GRPO) → Training (Megatron)       │
│       → 权重同步 (3D-HybridEngine) → 下一轮 AgentLoop                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### AgentLoop 对 Infra 各层的额外要求

| Infra 层 | 传统 LLM RL | AgentLoop (Agentic RL) | 新增需求 |
|----------|------------|----------------------|---------|
| **GPU** | 生成一次 response | 每个 sample 多轮 generate (5-10 轮) | 推理吞吐量需求 ×5-10 |
| **CPU** | 几乎不用 | 工具执行 (代码运行/API/DB)、环境管理 | 每节点 64+ 核 |
| **内存** | prompt + response | 多轮上下文累积 (可达 16K+ token) | KV Cache 更大 |
| **网络** | 训练通信 | 训练通信 + 推理 Server HTTP + 工具 API | HTTP 端口、外网访问 |
| **存储** | 模型 + checkpoint | + 沙箱临时文件、工具 corpus | Docker 镜像层 |
| **并发** | batch 内同步 | batch 内异步 (不同 sample 不同轮次数) | async + 动态批处理 |
| **容错** | 训练崩溃重启 | + 工具超时、沙箱崩溃、推理 Server 重连 | 超时/重试机制 |

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
| CPU | 64+ 核/节点 | Ray Worker 调度 + 数据预处理 + **AgentLoopWorker 并发** + 工具沙箱执行 |
| 内存 | 512GB+/节点 | Megatron 权重加载 + **多轮 KV Cache** + 数据缓冲区 |

**AgentLoop 对 CPU 的额外需求**：每个 AgentLoopWorker 是一个 Ray Actor (CPU 进程)，内部用 asyncio 并发运行多个 sample 的交互循环。默认配置 `agent.num_workers=8` 意味着 8 个 Worker 进程，每个 Worker 内部并发处理 `batch_size / num_workers` 个 sample。每个 sample 的工具调用 (如 `code_executor`) 又会 fork 子进程。因此：
- 保守估算：`num_workers × max_parallel_calls × 2` = 8 × 5 × 2 = **80 个并发进程**
- CPU 核数建议 ≥ AgentLoopWorker 数 + 工具并发数 + 系统开销

**AgentLoop 对内存的额外需求**：多轮交互中上下文不断累积（每轮追加 LLM 输出 + 工具返回），单条轨迹可达 16K-32K token。vLLM Server 需要为每个并发请求维护 KV Cache：
- KV Cache 估算：`并发请求数 × 上下文长度 × 模型维度 × 层数 × 2(K+V) × dtype_size`
- 236B MoE, 256 并发, 16K 上下文 ≈ 需要额外 **40-60GB** GPU 显存用于 KV Cache
- 通过 `gpu_memory_utilization: 0.85` 控制 KV Cache 上限

---

## 二、软件栈

### 核心依赖关系图

```
┌─────────────────────────────────────────────────────────┐
│                    你的训练代码                          │
│            (verl-agent-training/src/)                    │
│                                                         │
│  AgentLoop        Reward           Tool                 │
│  (agent_loop.py)  (reward_fn.py)   (tools/verl_tools.py)│
├─────────────────────────────────────────────────────────┤
│  verl v0.7.1+                                           │
│  ├─ AgentLoopBase        ← Rollout 编排                 │
│  ├─ AsyncLLMServerManager ← 推理服务代理                │
│  ├─ BaseTool              ← 工具接口                    │
│  ├─ compute_score         ← 奖励接口                    │
│  └─ DataProto             ← 数据协议                    │
├─────────────────────────────────────────────────────────┤
│  推理引擎              │  训练引擎       │  编排        │
│  vLLM ≥ 0.6            │  Megatron-LM    │  Ray ≥ 2.9   │
│  (Server Mode 必须)    │  (或 FSDP2)     │              │
├─────────────────────────────────────────────────────────┤
│  PyTorch ≥ 2.3         │  NCCL ≥ 2.20    │  CUDA ≥ 12.1 │
├─────────────────────────────────────────────────────────┤
│  NVIDIA Driver ≥ 535   │  Docker Engine   │  InfiniBand   │
│  (GPU 推理/训练)       │  (工具沙箱)      │  (集群通信)   │
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

## 四、AgentLoop 运行时 Infra

这是 Agentic RL 与传统 LLM RL 在 infra 上最本质的区别。AgentLoop 将 Rollout 从「一次生成」变为「多轮交互循环」，引入了推理服务、工具执行、上下文管理、并发控制等全新的 infra 需求。

### 4.1 推理服务层 — vLLM/SGLang Server Mode

AgentLoop **要求**推理引擎以 Server 模式运行（而非传统的 batch 模式），因为：
- 多轮交互需要在轮次之间保持 KV Cache（前缀缓存）
- 不同 sample 的交互轮次不同，需要动态批处理
- Sticky Session 需要路由同一个 sample 到同一个 Server 实例

```
┌─────────────────────────────────────────────────────────┐
│              vLLM/SGLang Server 集群                    │
│                                                         │
│  ┌─── Server 0 ───┐  ┌─── Server 1 ───┐    ┌─── N ───┐│
│  │  TP=8 (8 GPU)  │  │  TP=8 (8 GPU)  │    │ TP=8    ││
│  │  KV Cache Pool │  │  KV Cache Pool │    │         ││
│  │  动态批处理     │  │  动态批处理     │    │         ││
│  └────────────────┘  └────────────────┘    └─────────┘│
│           ▲                   ▲                 ▲      │
│           └───────────────────┼─────────────────┘      │
│                    GlobalRequestLoadBalancer            │
│                    (Sticky Session + 最少负载路由)      │
└─────────────────────────────────────────────────────────┘
```

**需要准备的 Infra：**

| 组件 | 配置 | 说明 |
|------|------|------|
| vLLM Server 实例数 | `total_gpus / rollout_tp` | 64 GPU, TP=8 → 8 个实例 |
| 每实例 GPU | `rollout_tp = 8` | DeepSeek MoE 至少 TP=8 |
| 每实例最大并发 | `max_connections: 1000` | 配置在 `rollout.server` |
| KV Cache 显存占比 | `gpu_memory_utilization: 0.85` | 剩余 15% 给模型权重 |
| 最大上下文长度 | `max_model_len: 16384` | 多轮累积可达 16K+ |
| Server 启动超时 | `max_start_wait_time: 300s` | 大模型加载慢 |
| 请求超时 | `timeout: 60s` | 单次 generate 超时 |
| 重试 | `max_attempts: 3, retry_delay: 2s` | Server 偶发超时重试 |

**Server Mode 必须的网络端口：**
- 每个 vLLM Server 实例需要一个 HTTP 端口（verl 自动分配）
- AgentLoopWorker 通过 HTTP 与 Server 通信
- 确保节点内 / 跨节点 HTTP 端口不被防火墙阻断

### 4.2 AgentLoopWorker — 并发执行架构

AgentLoopWorker 是 **CPU 侧的 Ray Actor**，负责编排每个 sample 的多轮交互：

```
AgentLoopManager
  │
  ├── AgentLoopWorker 0 (Ray Actor, CPU)
  │     ├── asyncio.gather(
  │     │     sample_0.run(),   # async 协程
  │     │     sample_1.run(),
  │     │     ...
  │     │     sample_31.run(),  # batch/num_workers 个协程
  │     │   )
  │     │
  │     │   每个协程内部:
  │     │     for turn in max_turns:
  │     │       await server_manager.generate()   # 异步等待 GPU
  │     │       tool_calls = parse(response)
  │     │       await tool.execute(tool_calls)     # 异步等待工具
  │     │       context += tool_response
  │     │
  │     └── 返回 List[AgentLoopOutput]
  │
  ├── AgentLoopWorker 1
  │     └── ...
  │
  └── AgentLoopWorker 7
        └── ...
```

**资源配置：**

| 参数 | 推荐值 | 配置位置 | 说明 |
|------|--------|---------|------|
| `num_workers` | 8 | `rollout.agent.num_workers` | AgentLoopWorker 数量 |
| `batch_size / num_workers` | 32 | 自动计算 | 每个 Worker 的并发 sample 数 |
| CPU per Worker | 4-8 核 | Ray 自动分配 | 协程 + 工具执行 |
| 内存 per Worker | 8-16 GB | Ray 自动分配 | 上下文缓存 + 工具状态 |

**关键设计决策 — Sticky Session：**

多轮交互中，同一个 sample 的多次 `generate()` 调用必须路由到同一个 vLLM Server 实例。原因：
- **前缀缓存 (Prefix Caching)**：第 N 轮的上下文包含前 N-1 轮的完整内容。如果路由到同一个 Server，前缀的 KV Cache 已经在显存中，只需要增量计算新 token。否则需要重新计算所有 token 的 KV。
- **性能影响**：非 Sticky 模式下，10 轮交互的计算量约为 Sticky 模式的 **5x**（前缀重复计算）。

verl 的 `GlobalRequestLoadBalancer` (Ray Actor) 自动实现 Sticky Session，按 `request_id` 路由。**这要求 Load Balancer 的 Ray Actor 网络可达所有 AgentLoopWorker 和 Server 实例。**

### 4.3 多轮上下文管理 — 内存与 Token 预算

AgentLoop 的上下文随轮次线性增长：

```
Turn 1:  [system + tools ~500] + [user ~200] + [LLM ~800]            ≈ 1,500 tokens
Turn 2:  + [tool_response ~500] + [LLM ~800]                         ≈ 2,800 tokens
Turn 3:  + [tool_response ~500] + [LLM ~800]                         ≈ 4,100 tokens
...
Turn 10: + [tool_response ~500] + [LLM ~800]                         ≈ 14,200 tokens
```

**Infra 影响：**

| 关注点 | 影响 | 对策 |
|--------|------|------|
| GPU 显存 | KV Cache 随上下文增长 | `max_model_len: 16384`, `gpu_memory_utilization: 0.85` |
| 推理延迟 | 长上下文 prefill 变慢 | Sticky Session 复用 KV Cache; Chunked Prefill |
| response_mask | 工具 response token 不参与梯度 | verl 自动处理，但需确保 `max_response_length` 足够 |
| 序列打包 | 变长轨迹难以高效打包 | verl 自动 pad，但浪费率可能较高 |

**Token 预算配置：**

```yaml
# 必须满足: max_prompt_length + max_response_length ≤ max_model_len
data:
  max_prompt_length: 2048           # system + tools + user
  max_response_length: 4096         # 多轮 LLM 输出 + tool responses (含 mask=0 部分)

actor_rollout_ref:
  rollout:
    max_new_tokens: 4096            # 与 max_response_length 一致
    tensor_model_parallel_size: 8
    gpu_memory_utilization: 0.85    # KV Cache 上限

    multi_turn:
      max_tool_response_length: 2048  # 单次工具返回截断长度 (防止工具返回过长)
```

**注意**：`max_response_length` 包含所有轮次的 LLM 输出和工具返回 token。如果 10 轮交互累积超过此值，AgentLoop 会强制终止。生产环境建议 `max_response_length ≥ 8192`。

### 4.4 工具执行 Infra

工具执行发生在 **CPU 侧**（AgentLoopWorker 进程内），但不同工具对 infra 的需求差异很大：

```
AgentLoopWorker (CPU)
  │
  ├─ calculator          → 进程内 eval()，无额外 infra    [~1ms]
  │
  ├─ code_executor       → subprocess (dev) 或 Docker (prod)
  │   │                     需要: Docker Engine, 沙箱镜像     [100ms-30s]
  │   └─ 并发: max_parallel_calls=5 个容器同时运行
  │
  ├─ web_search          → HTTP 请求到外部 API
  │   │                     需要: 外网访问, API Key           [200ms-5s]
  │   └─ mock 模式: 无外网需求
  │
  └─ database_query      → SQLite 连接 (进程内)
                            或远程 DB 连接                    [1ms-100ms]
```

**工具执行的 Infra 清单：**

| 工具 | 必须 | 推荐 | 生产环境 |
|------|------|------|---------|
| **code_executor** | Python 3.11 可用 | Docker Engine (隔离) | K8s Pod / Serverless |
| **web_search** | mock 模式 (无依赖) | SerpAPI Key | 自建搜索索引 |
| **database** | SQLite (无依赖) | 预构建任务 DB | PostgreSQL / MySQL |
| **通用** | 超时控制 (30s) | 资源限制 (CPU/内存) | 容器级隔离 + 网络隔离 |

**Docker 工具沙箱部署 (生产必须)：**

```bash
# 每个训练节点都需要:
# 1. Docker Engine 已安装并运行
systemctl status docker

# 2. 沙箱镜像已构建
docker build -t verl-sandbox:latest -f docker/Dockerfile.sandbox .

# 3. 验证隔离性
docker run --rm --network=none --read-only --memory=512m --cpus=1 \
    verl-sandbox:latest python3 -c "print('isolated')"

# 4. 确保训练进程有 docker 权限
#    (将运行训练的用户加入 docker 组，或使用 rootless Docker)
usermod -aG docker $TRAINING_USER
```

**工具执行并发控制：**

verl 的 `ToolAgentLoop` 通过 `max_parallel_calls` 控制每个 turn 内的并行工具调用数。但跨 sample 的并发由 asyncio 自然并行。在 64 GPU 集群上：

```
最大并发工具调用 = num_workers × (batch_size/num_workers) × max_parallel_calls
                 = 8 × 32 × 5
                 = 1,280 个并发工具调用 (理论极端值)

实际并发 (考虑工具调用比例和轮次分布):
  ≈ 50-200 个同时进行的工具调用

Docker 容器的系统级限制:
  - 确保 Docker daemon 的 max container 限制 ≥ 200
  - 确保 /tmp 有足够空间 (临时 Python 文件)
  - 确保 PID 限制足够 (每个容器 --pids-limit=64)
```

### 4.5 AgentLoop ↔ Training 数据流 Infra

AgentLoop 产出的数据通过 `DataProto` 流向 Training，关键在于 **response_mask**：

```
AgentLoop 输出:
  response_ids:  [tok tok tok ... tok tok tok ... tok tok tok ... pad pad]
  response_mask: [ 1   1   1  ...  0   0   0  ...  1   1   1  ...  0   0 ]
                   └─ LLM 生成 ─┘  └ tool response ┘  └─ LLM 生成 ─┘  └pad┘
                   参与策略梯度 ✓    不参与梯度 ✗       参与策略梯度 ✓   ✗

extra_fields:
  tool_rewards: [0.05, 0.05, 0.0]   ← 来自 BaseTool.execute() 的 step_reward
  num_turns: 5
```

**Infra 影响：**

- **DataProto 传输量增大**：多轮轨迹的 response_ids 比单轮长 3-10 倍，DataProto 的张量传输量相应增大
- **GPU 显存**：Training 阶段需要存储完整的 `response_ids`（含 tool response token），即使这些 token 被 mask 掉不参与梯度，仍占用显存
- **Reward 计算**：`compute_score()` 在 CPU 侧执行，接收完整的 `solution_str` 和 `extra_info`（含 `tool_rewards`），不消耗 GPU 资源

### 4.6 异步与 Off-Policy Infra

当使用异步训练模式（Rollout 和 Training 持续运行，不互相等待）时，AgentLoop 引入额外的 Off-Policy 挑战：

```
┌── Rollout (持续运行) ──┐    ┌── Training (持续运行) ──┐
│ AgentLoop 用 v1 权重    │    │ 用 v1 数据更新到 v2     │
│ 生成 10 轮交互          │    │                         │
│ (耗时 30s-2min)         │ ─→ │ 用 v2 数据更新到 v3     │
│                         │    │                         │
│ AgentLoop 完成时,       │    │ 训练已到 v3, 数据过期!  │
│ 训练可能已更新多个版本   │    │                         │
└─────────────────────────┘    └─────────────────────────┘
```

**对 infra 的要求：**

| 组件 | 说明 |
|------|------|
| **Trajectory Buffer** | 存储 AgentLoop 产出的轨迹，训练从中取数据 |
| **版本标记** | 每条轨迹记录生成时的模型版本号 |
| **丢弃策略** | 版本差距 > 2 的轨迹丢弃 (配置在 verl 的 async mode) |
| **Ray Object Store** | 缓冲区通常使用 Ray 的共享内存，确保节点内存充足 |

同步模式（默认推荐）下不需要这些，但吞吐量会受限于 AgentLoop 的长尾延迟。

### 4.7 容错与超时 Infra

AgentLoop 多轮交互引入了多个失败点，每个都需要对应的 infra 保障：

| 失败场景 | 表现 | Infra 保障 |
|----------|------|-----------|
| 工具执行超时 | code_executor 运行死循环 | `tool_timeout: 30s` + subprocess/Docker 超时 kill |
| 整条轨迹超时 | 10 轮交互总时间过长 | `rollout_timeout: 120s` → 强制终止，返回部分结果 |
| vLLM Server 崩溃 | GPU OOM 或 NCCL 错误 | `max_attempts: 3, retry_delay: 2s` + Server 自动重启 |
| Docker daemon 故障 | 容器无法启动 | 降级到 subprocess 模式 |
| 外部 API 不可用 | web_search / LLM-Judge 超时 | mock 模式降级 + 异步重试 |
| AgentLoopWorker 崩溃 | Ray Actor 异常退出 | Ray 自动重启 Actor |

**推荐的超时配置层次：**

```
rollout_timeout: 120s          ← 整条轨迹的总时限
  └─ per-turn generate: 60s    ← 单次 LLM 推理
  └─ per-tool execute: 30s     ← 单次工具调用
  └─ per-turn total: ~90s      ← 单轮 (generate + parse + execute)
```

---

## 五、DeepSeek MoE 特殊准备

这是最需要注意的部分，MoE 模型比 Dense 模型的 infra 要求高很多：

### 5.1 模型权重下载

```bash
# DeepSeek-V2.5 (236B MoE, ~470GB FP16 / ~235GB FP8)
huggingface-cli download deepseek-ai/DeepSeek-V2.5 --local-dir /shared/models/deepseek-v2.5

# 确保所有节点都能通过共享文件系统访问
```

### 5.2 Expert Parallelism (EP) 配置

MoE 模型需要 EP 来分散专家参数。verl + Megatron 的 EP 要求：
- EP 大小通常 = 节点内 GPU 数 (8) 或跨节点 (取决于网络)
- EP 通信走 All-to-All，对节点间带宽敏感
- **需要安装 DeepEP**（如果使用 FP8 MoE 推理加速）

### 5.3 路由一致性监控

训练和推理使用不同引擎 (Megatron vs vLLM)，MoE Router 可能选择不同专家。需要：
- 在训练配置中开启 `moe.monitor_routing_consistency: true`
- 如差异率 > 5%，考虑实现 R3 路由回放 (参考 MILES)

---

## 六、工具调用环境准备

> 注意: 工具执行的并发模型和容器化需求已在 §4.4 详述，此处聚焦安装部署步骤。

### 6.1 沙箱环境

```bash
# 构建沙箱 Docker 镜像 (所有训练节点)
docker build -t verl-sandbox:latest -f docker/Dockerfile.sandbox .

# 验证隔离性
docker run --rm --network=none --read-only --memory=512m \
    verl-sandbox:latest python3 -c "print('sandbox ok')"

# 验证并发 (模拟训练时的工具并发)
for i in $(seq 1 20); do
    docker run --rm --network=none verl-sandbox:latest \
        python3 -c "import time; time.sleep(1); print('done $i')" &
done
wait
echo "并发测试通过"
```

### 6.2 工具后端

| 工具 | 开发阶段 | 生产阶段 | verl 接口 |
|------|---------|---------|-----------|
| web_search | mock corpus | SerpAPI / Bing API (需 API Key) | `BaseTool.execute()` |
| code_executor | subprocess | Docker 容器 (每次调用独立容器) | `BaseTool.execute()` |
| calculator | 进程内 eval | 同 | `BaseTool.execute()` |
| database | SQLite in-memory | 预加载任务数据库 | `BaseTool.execute()` |

所有工具通过 `configs/tool_config.yaml` 注册到 verl 的 `ToolAgentLoop`：
```yaml
# tool_config.yaml 中每个工具指定:
- class_name: "src.tools.verl_tools.CalculatorTool"   # BaseTool 子类
  config: { type: native }                              # 传给 __init__ 的配置
  tool_schema: { type: function, function: { ... } }    # OpenAI 函数调用 schema
```

### 6.3 LLM-as-Judge (可选)

如果启用 LLM-as-Judge 奖励：
- 需要一个可用的 OpenAI-compatible API 端点
- 或部署一个独立的 Judge LLM (如 GPT-4 / Claude API)
- 预算：每 1000 条轨迹评估约 $5-15（GPT-4 价格）
- 在 `compute_score()` 中通过 `async` 调用实现，不阻塞 GPU 训练

---

## 七、分阶段落地路径

### 阶段 1 — 单机验证 (1-2 周)

```
硬件: 1 节点, 8× A100/H800
模型: Qwen2.5-7B-Instruct (Dense, 小模型)
训练: FSDP 后端 (无需 Megatron)
AgentLoop: ToolAgentLoop + 内置工具 (calculator, code_executor subprocess)
目标: 跑通 AgentLoop 多轮交互 + 工具调用 + compute_score 奖励 + GRPO 训练
验证重点:
  - response_mask 是否正确 (工具 response token 被 mask)
  - compute_score 的 extra_info["tool_rewards"] 是否传递正确
  - 多轮上下文是否正确累积
  - 工具超时和终止条件是否生效
```

这个阶段只需一台 8-GPU 服务器，验证全流程正确性。

### 阶段 2 — 小规模 MoE + Docker 沙箱 (2-3 周)

```
硬件: 2 节点, 16× H800
模型: DeepSeek-V2-Lite 或 Qwen2.5-MoE
训练: Megatron 后端, TP=8, PP=1, DP=2
AgentLoop: ToolAgentLoop + Docker 沙箱 (code_executor docker 后端)
目标: 验证 MoE EP 并行 + 权重同步 + Docker 工具隔离
验证重点:
  - Docker 容器并发执行工具的稳定性
  - 跨节点权重同步后 AgentLoop 能否正常使用新权重
  - vLLM Server Sticky Session 跨节点路由
```

### 阶段 3 — 全规模训练 (持续)

```
硬件: 8 节点, 64× H800
模型: DeepSeek-V2.5 (236B MoE)
训练: Megatron, TP=4, PP=2, DP=8, EP=8
AgentLoop: 8 个 vLLM Server 实例 (TP=8), 8 个 AgentLoopWorker
工具: Docker 沙箱 + 可选 web_search API + 可选 LLM-as-Judge
目标: 正式 Agentic RL 训练, 监控路由一致性和长尾延迟
关注指标:
  - rollout/num_turns_mean: 平均交互轮次 (目标 3-5)
  - rollout/timeout_ratio: 超时比例 (目标 < 5%)
  - reward/is_noop_ratio: 无工具调用比例 (目标持续下降)
  - system/weight_sync_ms: 权重同步延迟 (目标 < 500ms)
```

---

## 八、清单总结

| 类别 | 必须 | 推荐 | 可选 |
|------|------|------|------|
| **GPU** | 8× 80GB (验证) | 64× H800 (训练) | 128+ (生产) |
| **网络** | 节点内 NVLink | 节点间 IB 200Gbps | IB 400Gbps + GPUDirect RDMA |
| **存储** | NFS 2TB (模型) | Lustre 10TB (Checkpoint) | 高速 SSD Tier |
| **软件** | verl, vLLM, PyTorch, Ray | Megatron-LM, NCCL, WandB | DeepEP, NIXL |
| **沙箱** | subprocess (开发) | Docker (训练) | Kubernetes (生产) |
| **监控** | WandB | Prometheus + Grafana | 自定义 MoE 路由监控 |
| **AgentLoop** | ToolAgentLoop + tool_config.yaml | Sticky Session + Docker 工具沙箱 | 自定义 AgentLoop + 异步训练 |
| **奖励** | compute_score() 函数 | tool_rewards 步级奖励 | LLM-as-Judge (外部 API) |
| **并发** | asyncio 协程 | num_workers=8, max_parallel_calls=5 | 自定义并发策略 |
| **容错** | 工具超时 30s | rollout_timeout 120s + Server 重试 | 部分 rollout 回收 (APRIL) |

### AgentLoop Infra 核心检查清单

在启动训练前，确认以下每一项：

- [ ] vLLM/SGLang 可以 Server 模式启动并接受 HTTP 请求
- [ ] `tool_config.yaml` 中的每个工具类可以 import 且 `execute()` 可正常运行
- [ ] Docker daemon 运行中，`verl-sandbox:latest` 镜像已构建
- [ ] Docker 并发 ≥ 200 个容器不会触发系统限制 (`docker info` 检查)
- [ ] `max_prompt_length + max_response_length ≤ max_model_len`
- [ ] AgentLoopWorker 的 CPU/内存 Ray 资源充足
- [ ] 节点间 HTTP 端口不被防火墙阻断 (Server Mode 通信)
- [ ] `compute_score()` 函数可以独立运行并返回 `{"score": float}` 格式

最关键的瓶颈通常不是 GPU 数量，而是**节点间网络带宽**和**共享存储性能**。对于 AgentLoop，额外的瓶颈是**工具执行延迟**和**vLLM Server 的并发处理能力**——如果工具调用成为瓶颈，整个 Rollout 吞吐量会被拖慢。
