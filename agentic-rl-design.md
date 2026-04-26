# 基于 verl 的 Agentic RL 工具调用 Agent 训练方案 — 架构设计文档

> 日期: 2026-04-26 | 基座模型: DeepSeek MoE | 目标集群: 64+ GPU | 框架: verl v0.7.1+

---

## 1. 系统概述

### 1.1 目标

训练一个能够自主进行**多轮工具调用**的 LLM Agent，使其在面对复杂任务时能够：
- 正确解析任务意图并制定工具调用计划
- 通过 JSON 格式发起工具调用请求
- 根据工具返回结果继续推理或发起后续调用
- 在多轮交互后给出最终答案

### 1.2 技术选型理由

| 组件 | 选型 | 理由 |
|------|------|------|
| RL 框架 | verl v0.7.1+ | 最成熟的开源 LLM RL 框架，EuroSys 2025，支持 AgentLoop 多轮交互 |
| 基座模型 | DeepSeek-V2.5/V3 (MoE) | MoE 架构推理高效，已验证 671B 规模 RL 训练 |
| 推理引擎 | vLLM Server Mode | DeepSeek MoE 支持好，动态批处理，FP8 加速 |
| 训练引擎 | Megatron-LM | 64+ GPU 生产推荐，5D 并行，EP 支持 |
| RL 算法 | GRPO (主) / DAPO (备) | 无需 Critic 模型，节省 ~50% GPU 资源 |
| 编排 | Ray | 单控制器编排，placement group 管理 |

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     RLTrainer (单控制器 — verl HybridFlow)              │
│                                                                         │
│  ┌───────────┐   ┌───────────────┐   ┌───────────┐   ┌──────────────┐  │
│  │  Prompt    │   │   Rollout     │   │  Reward   │   │  Training    │  │
│  │  Dispatch  │──▶│  (AgentLoop)  │──▶│  Compute  │──▶│  Update      │  │
│  │            │   │               │   │           │   │  (GRPO/DAPO) │  │
│  └───────────┘   └───────────────┘   └───────────┘   └──────────────┘  │
│       │                │                   │                │           │
└───────┼────────────────┼───────────────────┼────────────────┼───────────┘
        │                │                   │                │
        ▼                ▼                   ▼                ▼
  ┌───────────┐   ┌────────────────┐   ┌──────────┐   ┌──────────────┐
  │ DataProto │   │ vLLM Server   │   │ Sandbox  │   │ Megatron-LM  │
  │ 数据协议   │   │ (FP8推理)     │   │ 环境管理  │   │ 5D并行训练    │
  └───────────┘   └────────────────┘   └──────────┘   └──────────────┘
                        │                   │
                        ▼                   ▼
                  ┌────────────┐     ┌─────────────┐
                  │ 工具执行    │     │ 结果验证     │
                  │ API/DB/Code│     │ Rule+Judge  │
                  └────────────┘     └─────────────┘
```

### 2.2 数据流概览

```
Prompt Dataset
    │
    ▼
┌─ DataProto 分片分发 ─────────────────────────────────────────┐
│                                                               │
│  ┌──────────────── AgentLoop (每个 Prompt) ──────────────┐   │
│  │                                                        │   │
│  │  Turn 1: LLM 生成 → 解析工具调用 → Sandbox 执行       │   │
│  │      ↓                                                 │   │
│  │  Turn 2: 拼接结果 → LLM 继续推理 → 可能再次调用       │   │
│  │      ↓                                                 │   │
│  │  ...                                                   │   │
│  │      ↓                                                 │   │
│  │  Turn N: 任务完成 / 达到最大轮次                        │   │
│  │                                                        │   │
│  │  输出: 完整轨迹 + 每步 log-prob + 工具调用记录         │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                               │
│  Reward 计算 (Rule-Based + LLM-Judge)                        │
│      ↓                                                        │
│  Ref Log-prob 计算 (参考策略)                                 │
│      ↓                                                        │
│  Advantage 估计 (GRPO 组内相对优势)                           │
│      ↓                                                        │
│  Training 更新 (GRPO Clip + 梯度)                            │
│      ↓                                                        │
│  权重同步 (3D-HybridEngine: PP → micro-DP 重分片)            │
│      ↓                                                        │
│  下一轮 Prompt 分发                                           │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. 训练流程 (6 步循环)

### Step ① Prompt 分发

- 从训练数据集加载 Prompt，注入工具描述 (System Prompt + Tool Schema)
- 使用 `DataProto` 分片分发到各 Rollout Worker
- GRPO 模式: `repeat-chunk-dispatch`，每个 Prompt 重复 `group_size` 次

### Step ② Rollout 生成 (AgentLoop 多轮)

```
for each prompt in batch:
    context = system_prompt + tool_descriptions + user_query
    trajectory = []
    for turn in range(max_turns):
        response = vllm_generate(context)        # 记录 log-prob
        tool_calls = parse_tool_calls(response)
        if no tool_calls:
            trajectory.append(response)
            break
        results = sandbox.execute(tool_calls)     # 并行执行工具
        context += response + format_results(results)
        trajectory.append((response, tool_calls, results))
    collect(trajectory, log_probs)
```

关键设计:
- **动态批处理**: 不同任务的工具调用次数不同，vLLM Server Mode 天然支持
- **长尾处理**: 超时 120s 的 Rollout 强制终止，回收已生成的部分结果（参考 SLIME APRIL 思路）
- **并发控制**: 每个 vLLM 实例最多 256 并发请求

### Step ③ Reward 计算

```python
def compute_reward(prompt, trajectory, ground_truth):
    # 1. 格式奖励: 工具调用 JSON 是否合法
    format_score = check_format(trajectory)           # +0.1 / -0.1

    # 2. 任务奖励: 最终答案是否正确
    task_score = verify_answer(trajectory, ground_truth)  # +1.0 / 0.0

    # 3. 过程奖励: 每次正确工具调用
    process_score = count_valid_tool_calls(trajectory) * 0.05

    # 4. LLM-as-Judge (异步): 推理质量评估
    judge_score = llm_judge(prompt, trajectory)       # 0.0 ~ 1.0

    # 5. No-op 惩罚: 无工具调用直接回答
    if no_tool_calls_in(trajectory):
        return (task_score * 0.5 + format_score) * 0.3

    return (0.7 * (task_score + format_score + process_score)
          + 0.3 * judge_score)
```

### Step ④ Ref Log-prob 计算

- 使用冻结的参考策略 (初始模型权重) 计算每个 Token 的 log-prob
- 用于 KL 散度约束: `KL(π_θ || π_ref)`
- 参考模型与当前模型共置在同一 GPU，交替计算

### Step ⑤ Advantage 估计 (GRPO)

```python
# GRPO: 组内相对优势
for group in prompt_groups:  # 每组 group_size=8 个 response
    rewards = [r for r in group.rewards]
    mean_r = mean(rewards)
    std_r = std(rewards)
    for i, response in enumerate(group):
        response.advantage = (rewards[i] - mean_r) / (std_r + 1e-8)
```

- 无需 Critic 模型，节省 ~50% GPU 资源
- 组内归一化消除绝对奖励尺度的影响
- 如遇熵坍塌: 切换到 DAPO (Dynamic Sampling + Clip-Higher + Token-Level Loss)

### Step ⑥ Training 更新

```python
# GRPO / PPO-style Clip 更新
ratio = exp(log_prob - old_log_prob)
clipped = clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
policy_loss = -min(ratio * advantage, clipped * advantage)
kl_penalty = beta * KL(π_θ || π_ref)
loss = policy_loss + kl_penalty
optimizer.step()
```

- Megatron 5D 并行: DP × TP × PP × CP × EP
- 梯度累积: micro_batch_size 适配显存
- 混合精度: BF16 (训练) + FP8 (推理)

**权重同步**: 3D-HybridEngine 将训练并行 (PP 维度) 重分片为推理并行 (micro-DP)，NCCL 广播，延迟 < 300ms。

---

## 4. 核心组件设计

### 4.1 Rollout 层 — AgentLoop + vLLM Server

#### 架构

```
┌──────────────────────────────────────────────────────┐
│                  verl AgentLoop                       │
│                                                      │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐ │
│  │ Prompt   │────▶│ vLLM     │────▶│ Tool Call    │ │
│  │ Manager  │     │ Server   │     │ Parser       │ │
│  └──────────┘     └──────────┘     └──────────────┘ │
│       ▲                                    │         │
│       │                                    ▼         │
│  ┌──────────┐                      ┌──────────────┐ │
│  │ Context  │◀─────────────────────│ Sandbox      │ │
│  │ Builder  │                      │ Executor     │ │
│  └──────────┘                      └──────────────┘ │
└──────────────────────────────────────────────────────┘
```

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 推理引擎 | vLLM Server Mode | DeepSeek MoE 支持好 |
| 量化 | FP8 (W8A8) | 推理加速，显存节省 |
| max_new_tokens | 4096 | 每轮最大生成长度 |
| max_turns | 10 | AgentLoop 最大交互轮次 |
| tool_timeout | 30s | 单次工具调用超时 |
| rollout_timeout | 120s | 整个 Rollout 超时 |
| temperature | 0.7 | 采样温度 |
| top_p | 0.95 | Nucleus 采样 |

#### 多轮交互协议

LLM 输出工具调用时使用标准 JSON 格式:

```json
<tool_call>
{"name": "web_search", "arguments": {"query": "verl framework documentation"}}
</tool_call>
```

工具执行结果以 `<tool_response>` 标签注入上下文:

```json
<tool_response>
{"name": "web_search", "result": "verl is a flexible RL training framework..."}
</tool_response>
```

### 4.2 环境层 — 工具沙箱

#### 沙箱架构

```
┌─────────────────────────────────────────┐
│            ToolEnvironment               │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │        Tool Registry             │   │
│  │  ┌──────┐ ┌──────┐ ┌──────────┐ │   │
│  │  │Search│ │Calc  │ │CodeExec  │ │   │
│  │  └──────┘ └──────┘ └──────────┘ │   │
│  │  ┌──────┐ ┌──────┐              │   │
│  │  │  DB  │ │Custom│              │   │
│  │  └──────┘ └──────┘              │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │      Execution Sandbox           │   │
│  │  • Docker 容器隔离               │   │
│  │  • 超时控制 (30s)                │   │
│  │  • 资源限制 (CPU/Memory)         │   │
│  │  • 网络隔离 (可配置)              │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │      State Manager               │   │
│  │  • Episode 前环境清理            │   │
│  │  • 中间产物跟踪                  │   │
│  │  • 防泄露检查 (参考 ROLL)        │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 内置工具集

| 工具 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `web_search` | 网页搜索 | query: str | results: List[dict] |
| `calculator` | 数学计算 | expression: str | result: float |
| `code_executor` | Python 执行 | code: str | stdout + return_value |
| `database_query` | SQL 查询 | sql: str, db: str | rows: List[dict] |

#### 工具注册接口

```python
@tool_registry.register
class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information"
    parameters = {
        "query": {"type": "string", "description": "Search query"}
    }

    def execute(self, query: str) -> dict:
        ...
```

### 4.3 Training 层 — Megatron + GRPO

#### 并行策略 (64 GPU 基线)

```
8 节点 × 8 GPU (H800 80GB)

Megatron 5D 并行:
  TP (Tensor Parallel)     = 4    # 节点内
  PP (Pipeline Parallel)   = 2    # 跨节点
  DP (Data Parallel)       = 8    # 自动计算: 64 / (TP × PP) = 8
  EP (Expert Parallel)     = 8    # MoE 专家并行
  CP (Context Parallel)    = 1    # 按需启用 (长上下文时)
```

#### GRPO 算法配置

| 参数 | 值 | 说明 |
|------|-----|------|
| group_size | 8 | 每个 Prompt 生成 8 个 Response |
| clip_ratio | 0.2 | PPO-style 裁剪比率 |
| kl_coeff | 0.01 | KL 散度惩罚系数 |
| entropy_coeff | 0.01 | 熵正则化系数 |
| learning_rate | 1e-6 | 学习率 |
| warmup_steps | 50 | 预热步数 |
| max_grad_norm | 1.0 | 梯度裁剪 |
| epochs_per_update | 1 | 每批数据训练轮次 |

#### DAPO 备选方案 (熵坍塌时切换)

- **Dynamic Sampling**: 过滤全对/全错的 group，聚焦高方差样本
- **Clip-Higher**: 正优势使用更大裁剪范围 (1 + 0.28)，负优势使用更小 (1 - 0.2)
- **Token-Level Loss**: 不按序列长度归一化，避免长回答被惩罚
- **Overlong Penalty**: 超过上下文窗口的回答给予 -1 奖励

### 4.4 奖励层 — 混合奖励系统

#### 奖励组成

```
Total Reward = w_rule × R_rule + w_judge × R_judge

其中:
  R_rule  = R_format + R_task + R_process    (确定性, 低延迟)
  R_judge = LLM-as-Judge 评分                 (概率性, 高延迟)

默认权重:
  w_rule  = 0.7
  w_judge = 0.3
```

#### 各奖励分量

| 分量 | 范围 | 计算方式 |
|------|------|---------|
| R_format | [-0.1, +0.1] | 工具调用 JSON 格式校验 |
| R_task | [0, 1.0] | 最终答案与 ground truth 匹配 |
| R_process | [0, 0.5] | 每次有效工具调用 +0.05 (上限 10 次) |
| R_judge | [0, 1.0] | GPT-4/Claude 评估推理质量 |
| No-op penalty | ×0.3 | 无工具调用时总奖励打 3 折 |

#### 奖励工程注意事项

1. **奖励尺度**: GRPO 组内归一化会消除绝对尺度影响，但各分量的相对权重仍需调优
2. **Judge 异步**: LLM-as-Judge 延迟高 (1-5s)，使用异步批处理，不阻塞训练
3. **伪阳性处理**: 参考 ROLL，引入多 LLM 交叉验证，过滤不可靠的正向奖励
4. **No-op 检测**: 防止模型学会不调用工具直接回答以获得部分奖励

### 4.5 数据层 — DataProto + 异步

#### DataProto 统一协议

```python
# verl DataProto 核心字段
DataProto:
    batch:
        input_ids: Tensor[B, S]        # 输入 Token
        attention_mask: Tensor[B, S]   # 注意力掩码
        log_probs: Tensor[B, S]        # 采样 log-prob
        ref_log_probs: Tensor[B, S]    # 参考 log-prob
        advantages: Tensor[B]          # 优势值
        rewards: Tensor[B]             # 奖励值
    meta_info:
        prompt_ids: List[str]          # Prompt 标识
        tool_calls: List[List[dict]]   # 工具调用记录
        num_turns: List[int]           # 交互轮次数
        model_version: int             # 模型版本号
```

#### 异步训练模式

```
┌─────────────┐         ┌──────────┐         ┌─────────────┐
│  Rollout    │  ──▶    │  Buffer  │  ──▶    │  Training   │
│  Workers    │         │  (版本   │         │  Workers    │
│  (持续生成) │         │   管理)  │         │  (持续训练) │
└─────────────┘         └──────────┘         └─────────────┘

Off-Policy 控制:
  - 每条轨迹记录生成时的模型版本
  - 版本差距 > 2 的数据丢弃
  - 可选: 重要性采样 (IS) 修正 off-policy 偏差
```

### 4.6 I/O 层 — 权重同步与通信

#### 通信架构

| 层级 | 机制 | 延迟 | 用途 |
|------|------|------|------|
| 训练内部 | NCCL AllReduce | ~ms | 梯度聚合 |
| 训练↔推理 | 3D-HybridEngine | <300ms | 权重重分片同步 |
| 控制平面 | Ray RPC | ~ms | 调度、状态管理 |
| 推理服务 | HTTP/gRPC | ~ms | vLLM Server API |
| 工具调用 | HTTP/subprocess | 1-30s | Sandbox 执行 |

#### 3D-HybridEngine 权重同步

```
训练并行布局:        推理并行布局:
  TP=4, PP=2           TP=8, DP=auto

重分片流程:
  1. PP 维度收集 (AllGather pipeline stages)
  2. 转换为 micro-DP 布局
  3. NCCL Broadcast 到 vLLM Server
  4. vLLM 热更新权重 (UpdateWeightFromDistributed)
```

---

## 5. DeepSeek MoE 专项考量

### 5.1 训练-推理路由不一致问题

**问题**: MILES R3 的实证发现表明，SGLang/vLLM 推理与 Megatron 训练中约 10% 的 Router 选择不同专家，94% 的 Token 至少在一层有不同路由。

**影响**: Rollout 阶段计算的 log-prob 与 Training 阶段不一致，引入策略梯度估计偏差。

**解决方案 (分级)**:

| 级别 | 方案 | 开销 | 适用场景 |
|------|------|------|---------|
| L0 | 忽略差异，监控指标 | 零 | 初始实验 |
| L1 | 统一精度 (FP8 推理 + BF16 训练) | 低 | 精度敏感场景 |
| L2 | R3 路由回放 (记录推理路由，训练时重放) | 中 | 严格一致性需求 |
| L3 | 统一引擎 (训练和推理使用同一引擎) | 高 | 极端一致性需求 |

**推荐**: 先用 L0 启动，监控路由差异率。如差异率 > 5% 且影响收敛，升级到 L2。

### 5.2 MoE 并行策略

```
DeepSeek-V2.5 (236B MoE, 21B 活跃参数):
  64 GPU 配置:
    TP=8 (单节点 8 GPU)
    PP=2 (2 个 Pipeline Stage)
    DP=4 (4 路数据并行)
    EP=8 (8 路专家并行)

DeepSeek-V3 (671B MoE, 37B 活跃参数):
  64 GPU 配置:
    TP=8 (单节点 8 GPU)
    PP=4 (4 个 Pipeline Stage)
    DP=2 (2 路数据并行)
    EP=8 (8 路专家并行)
  注: 显存紧张时启用 LoRA (rank=64)
```

### 5.3 Expert 负载均衡

- **Auxiliary Loss**: 添加 Load Balancing Loss (系数 0.01) 防止专家负载不均
- **容量因子**: 每个专家的容量上限设为 1.5 × 平均负载
- **监控**: 训练中持续记录每个专家的 Token 分配比例

---

## 6. 集群部署方案

### 6.1 硬件需求

| 组件 | 规格 | 数量 |
|------|------|------|
| GPU | H800 80GB / A100 80GB | 64+ |
| 节点 | 8 GPU / 节点 | 8+ |
| 网络 | InfiniBand 400Gbps / RoCE v2 | 全连接 |
| 存储 | 共享 NFS/Lustre | ≥10TB |
| CPU | 64 核 / 节点 | 与 GPU 节点相同 |
| 内存 | 512GB / 节点 | 与 GPU 节点相同 |

### 6.2 部署模式对比

#### 模式 A: 共置模式 (GPU 复用，推荐)

```
┌────────────────────────────────────────────────────┐
│  8 节点 × 8 GPU = 64 GPU                          │
│                                                    │
│  所有 GPU 同时承担训练和推理任务                   │
│  3D-HybridEngine 在训练/推理间切换权重             │
│                                                    │
│  优点: GPU 利用率高，无资源浪费                    │
│  缺点: 训练和推理串行，不能流水线并行              │
│                                                    │
│  适用: 实验阶段，GPU 资源有限                      │
└────────────────────────────────────────────────────┘
```

#### 模式 B: 解耦模式 (独立资源池)

```
┌──────────────────┐  ┌──────────────────────────────┐
│  Rollout 资源池    │  │  Training 资源池              │
│                    │  │                              │
│  2 节点 (16 GPU)  │  │  6 节点 (48 GPU)             │
│  vLLM Server      │  │  Megatron-LM                 │
│  TP=8, 2 实例     │  │  TP=4, PP=2, DP=6            │
│  FP8 推理         │  │  BF16 训练                    │
│                    │  │                              │
│  优点: 训练推理并行│  │  优点: 各自独立优化            │
│  缺点: 需要跨节点  │  │  缺点: 资源利用率可能不均     │
│        权重同步    │  │                              │
└──────────────────┘  └──────────────────────────────┘

适用: 生产训练，需要最大吞吐量
```

#### 模式 C: 异步模式 (最大吞吐，需更多 GPU)

```
┌──────────────────┐  ┌──────────────────┐  ┌────────────┐
│  Rollout Pool    │  │  Training Pool   │  │  Tool Pool │
│  16 GPU          │  │  48 GPU          │  │  CPU Only  │
│  持续生成轨迹     │  │  持续训练更新     │  │  沙箱执行   │
│  异步权重更新     │  │  从 Buffer 取数据 │  │  Docker    │
└──────────────────┘  └──────────────────┘  └────────────┘
         │                    ▲
         ▼                    │
    ┌──────────────────────────┐
    │  Trajectory Buffer       │
    │  (版本管理 + Off-Policy) │
    └──────────────────────────┘

适用: 大规模生产训练，GPU 充足
```

### 6.3 推荐部署策略

**阶段 1 — 验证 (单机 8 GPU)**:
- 使用 Qwen2.5-7B (Dense 模型)
- FSDP 后端，共置模式
- 验证 AgentLoop + 工具调用 + 奖励函数

**阶段 2 — 扩展 (64 GPU)**:
- 切换到 DeepSeek-V2.5 (236B MoE)
- Megatron 后端，共置模式 (模式 A)
- 调优并行策略和超参

**阶段 3 — 生产 (128+ GPU)**:
- 解耦模式 (模式 B) 或异步模式 (模式 C)
- 完整监控 + 自动断点恢复
- 引入 DAPO 防止熵坍塌

---

## 7. 监控与调优

### 7.1 关键指标

| 类别 | 指标 | 目标值 |
|------|------|--------|
| 训练 | Policy Loss | 持续下降 |
| 训练 | KL 散度 | 0.01-0.1 |
| 训练 | Entropy | 保持适当高 (防坍塌) |
| 训练 | Gradient Norm | < max_grad_norm |
| Rollout | 平均轮次数 | 2-5 (任务相关) |
| Rollout | 工具调用成功率 | > 90% |
| Rollout | 生成吞吐量 (tok/s) | 持续监控 |
| 奖励 | 平均奖励 | 持续上升 |
| 奖励 | 奖励方差 | 组内方差不为零 |
| MoE | 路由一致率 | > 90% |
| MoE | 专家负载均衡度 | 标准差 < 0.1 |
| 系统 | GPU 利用率 | > 70% |
| 系统 | 权重同步延迟 | < 500ms |

### 7.2 常见问题与解决

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 熵坍塌 | 所有输出趋同 | 切换 DAPO + 增大 entropy_coeff |
| 奖励 hacking | 奖励上升但质量下降 | 检查奖励函数，增加 Judge 权重 |
| 训练不稳定 | Loss 震荡 | 降低 lr，增大 clip_ratio |
| Rollout 太慢 | GPU 利用率低 | 增大批量，检查工具调用延迟 |
| OOM | GPU 显存不足 | 减小 micro_batch，启用 LoRA |
| 路由不一致 | KL 异常 | 启用 R3 路由回放 |

---

## 8. 与其他系统的对比

| 特性 | 本方案 (verl) | ROLL | SLIME | Forge |
|------|-------------|------|-------|-------|
| 多轮 Agent | AgentLoop | AgentServer + Chunked MDP | OpenClaw-RL | Agent 多轮 |
| MoE 支持 | Megatron EP | DeepSpeed + Megatron | Megatron + DeepEP | 未公开 |
| 路由一致性 | 监控 + 可选 R3 | 无特殊处理 | 无特殊处理 | 无特殊处理 |
| Off-Policy | 版本管理 + 丢弃 | Chunked MDP IS | 双边 IS | Windowed FIFO |
| 长尾处理 | 超时回收 | RollPacker | APRIL | Windowed FIFO |
| 环境管理 | Docker 沙箱 | Rock + iFlow | 外部 Agent | 内建 |
| 异步训练 | 支持 | 支持 | 支持 | 支持 |
| 最大验证规模 | 671B MoE | 3000+ GPU | 355B MoE (64 GPU) | 未公开 |

---

## 9. 项目文件说明

详见 `verl-agent-training/` 目录:

| 文件 | 说明 |
|------|------|
| `configs/grpo_deepseek_tool.yaml` | 主训练配置 (GRPO + DeepSeek + 工具调用) |
| `configs/ppo_deepseek_tool.yaml` | PPO 备选配置 |
| `scripts/train.sh` | 训练启动脚本 |
| `scripts/setup_cluster.sh` | 集群环境初始化 |
| `scripts/eval.sh` | 评估脚本 |
| `src/reward_function.py` | 混合奖励函数 |
| `src/tool_env.py` | 工具沙箱环境 |
| `src/agent_loop.py` | AgentLoop 多轮交互封装 |
| `src/data_processor.py` | 数据预处理 |
| `src/tools/` | 内置工具集 |
| `data/` | 示例训练/评估数据 |
| `docker/Dockerfile.sandbox` | 沙箱 Docker 镜像 |
