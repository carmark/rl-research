# 基于 verl 的 Agentic RL 工具调用 Agent 训练方案 — 架构设计文档

> 日期: 2026-04-27 | 基座模型: DeepSeek MoE | 目标集群: 64+ GPU | 框架: verl v0.7.1+

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
| 基座模型 | DeepSeek-V4-Flash (284B, 13B活跃) / V2.5 (兼容) | V4-Flash 百万Token上下文, CSA/HCA 注意力, FP4 量化, OPD 训练范式 |
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

#### 4.1.1 PD 解耦 — Prefill-Decode 分离 (参考 SLIME/GLM-5)

多轮 Agentic RL 场景下，长前缀 Prefill (对话历史、工具轨迹、代码上下文) 与 Decode 在同一资源上混合运行会造成严重干扰——重 Prefill 会抢占/中断正在进行的 Decode，导致尾部延迟恶化。

**解决方案**: 将 Prefill 和 Decode 分离到专用资源:

```
共享模式 (实验/小规模):              生产模式 (PD 解耦):
┌──────────────────────┐           ┌─── Prefill 资源池 ───┐  ┌─── Decode 资源池 ───┐
│  vLLM Server         │           │  Prefill Worker ×N    │  │  Decode Worker ×M    │
│  Prefill + Decode    │  ──→      │  专注长前缀计算       │  │  稳定不中断          │
│  混合运行, 互相干扰   │           │  (GPU 计算密集)       │  │  (显存带宽密集)      │
└──────────────────────┘           └──────────────────────┘  └─────────────────────┘
```

| 部署方案 | 适用场景 | 特点 |
|---------|---------|------|
| 共享模式 (关闭 PD 分离) | 验证阶段、短上下文 | 简单，无额外通信开销 |
| 生产模式 (PD 独立节点) | 多轮 Agent 训练、200K+ 上下文 | Decode 稳定，尾部延迟显著改善 |

#### 4.1.2 DP-aware Routing (参考 SLIME/GLM-5)

现有 Sticky Session 基于 `request_id` 将同一请求路由到同一 Server，但在多轮 Agentic 场景下，同一 Agent 实例（一个完整 Rollout）会发起多个请求，每个请求有不同 `request_id`。

**升级**: 从 Request 级亲和性 → **Rollout 级亲和性 (DP-aware)**:

- 同一 Agent 实例 (Rollout) 的**所有**请求路由到同一 SGLang/vLLM Server
- 最大化 KV Cache 复用 (连续请求共享前缀)
- 在 Data Parallelism 下保持 KV Cache 局部性

```
传统 Sticky Session:                DP-aware Routing:
  req_1 → Server_A                    rollout_1 (req_1,2,3,...) → Server_A  ← 全部请求
  req_2 → Server_B  ← 不同 Server     rollout_2 (req_4,5,6,...) → Server_B  ← 全部请求
  req_3 → Server_A                    rollout_3 (req_7,8,9,...) → Server_C  ← 全部请求
```

配置: 在 `GlobalRequestLoadBalancer` 中以 `rollout_id` (而非 `request_id`) 作为路由键。

#### 4.1.3 全局 KV Cache Pool (参考 Seer/Mooncake)

Chunk 级 Divided Rollout (见 §4.1.4) 要求 KV Cache 可在推理实例间迁移。单次 Rollout 迭代可产生**数十 TB** KV Cache，远超单实例 DRAM 容量。

**方案**: 基于 Mooncake 构建跨推理节点的全局共享 KV Cache Pool:

```
┌──── 推理实例 1 ────┐  ┌──── 推理实例 2 ────┐  ┌──── 推理实例 N ────┐
│  本地 KV Cache      │  │  本地 KV Cache      │  │  本地 KV Cache      │
│  (GPU HBM, 热层)    │  │  (GPU HBM, 热层)    │  │  (GPU HBM, 热层)    │
└────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Mooncake 全局 KV Cache Pool                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  DRAM 热层    │  │  NVMe SSD    │  │  分布式存储   │              │
│  │  (高频访问)   │  │  (温层)      │  │  (冷层)       │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

- **迁移时无需重新 Prefill**: chunk 级调度将请求迁移到其他实例时，直接从 Pool 读取 KV Cache
- **容量估算**: Kimi-K2 场景 (平均输出 39K token, DP32+EP32)，单次迭代 KV Cache 约 10-50 TB
- **实现路径**: 本地 KV Cache → 3FS 共享 → Mooncake 两层缓存

#### 4.1.4 Chunk 级 Divided Rollout (参考 Seer)

**问题量化**: RL Rollout 的输出长度分布极度不均，长尾效应严重。Seer 的生产数据显示:
- 尾部 10% 的请求占 50% 的总时间
- 尾部延迟是平均值的 5-20 倍

传统方案将整个 GRPO 组绑定到单个实例，导致实例间负载严重不均。

**Divided Rollout**: 将请求分解为 chunk 级可调度单元:

```
传统 Rollout:                      Divided Rollout:
┌─ Instance A ──────────────┐     ┌─ Instance A ─┐  ┌─ Instance B ─┐
│ Group 1: ████████████████ │     │ G1-chunk1 ███│  │ G1-chunk3 ███│
│ (长 Group, 独占实例)       │     │ G2-chunk1 ██ │  │ G1-chunk4 ██ │
├─ Instance B ──────────────┤     │ G3-chunk2 ███│  │ G2-chunk2 ███│
│ Group 2: ████             │     └──────────────┘  └──────────────┘
│ (短 Group, 早早完成空闲)   │
└───────────────────────────┘      → 持续再平衡, 尾部延迟 -72~94%
```

**Context-Aware Scheduling (双队列 + 投机探针)**:
1. 高优先级队列: 投机探针请求 (SFS 最短优先), 用于估计 Group 长度
2. 低优先级候选集: LFS 最长优先调度
3. 保守更新: 取最长观测值, 防止饥饿
4. 效果: 接近拥有完美长度信息的 Oracle LFS (仅差 7%)

**实现前提**: 全局 KV Cache Pool (§4.1.3)

#### 4.1.5 DGDS 投机解码 (参考 Seer, 可选)

DGDS (Distributed Grouped Draft Server) 利用 GRPO 组内 Response 的 Token 模式相似性加速解码，**无需独立草稿模型**:

- **组内 CST (Compressed Suffix Tree)**: 聚合同组请求的 Token 更新，构建压缩后缀树作为草稿来源
- **MBA (Marginal-Benefit-Aware) 自适应**: 动态计算最优草稿长度，高优先级请求 (探针) 获得更大 draft budget
- **长尾阶段特殊优化**: 并发度低时自动增加草稿深度, 启用多路径草稿 (top-k branching)

| 场景 | 启用建议 | 预期加速 |
|------|---------|---------|
| 正常 batch | 可选 | +26-48% |
| 长尾阶段 (并发度低) | 推荐 | +54% (多路径 k=4) |
| 短序列 / 小 group | 不推荐 | 通信开销可能抵消收益 |

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

#### DSec 启发的执行基底扩展 (规划中)

参考 DeepSeek V4 的 DSec 沙箱平台，计划扩展执行基底：
- **Bare Container**: 当前 Docker 沙箱即等价实现
- **Browser Container**: Headless Chrome，用于 Web Agent 训练
- **分层存储**: 热层 NVMe + 冷层 HDD，加速镜像加载
- **轨迹日志**: 完整记录环境交互用于回放调试
- **可抢占恢复**: 沙箱状态检查点 + 自动恢复

#### Multi-Task Rollout Orchestrator (参考 SLIME/GLM-5)

多任务 Agentic RL 中，不同任务依赖不同工具集、不同 Rollout 逻辑、不同环境配置。GLM-5 通过微服务架构解决异构轨迹生成问题:

```
┌────────────────────────────────────────────────────────────────┐
│                  Multi-Task Rollout Orchestrator                │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ SWE 任务  │  │ Terminal │  │ Search   │  │ Custom       │  │
│  │ Rollout   │  │ Rollout  │  │ Rollout  │  │ Rollout      │  │
│  │ +Reward   │  │ +Reward  │  │ +Reward  │  │ +Reward      │  │
│  │ (微服务)  │  │ (微服务)  │  │ (微服务)  │  │ (微服务)     │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │
│       │              │             │              │            │
│       └──────────────┼─────────────┼──────────────┘            │
│                      ▼                                         │
│              中央编排器: 控制采样比率、动态调整、进度监控         │
│                      │                                         │
│                      ▼                                         │
│         标准化 message-list 轨迹表示 (联合训练)                 │
└────────────────────────────────────────────────────────────────┘
```

关键设计:
- **即插即用**: 每个任务实现独立的 Rollout + Reward 逻辑作为微服务，新任务无需修改核心训练循环
- **中央编排器**: 控制每任务 Rollout 比率和生成速度，实现均衡数据收集
- **标准化轨迹**: 所有 Agentic 任务轨迹统一为 message-list 格式，支持异构工作负载联合训练
- **生产规模**: 已验证支持 1000+ 并发 Rollout

#### 10K+ 可验证环境构建 (参考 GLM-5)

GLM-5 在 Agentic RL 阶段构建了 10K+ 可验证环境，覆盖三大类:

| 环境类型 | 规模 | 特点 |
|---------|------|------|
| SWE 环境 (代码修复) | 数千实例 | RepoLaunch 管线, 自动化测试验证 |
| Terminal 环境 (终端任务) | 数千实例 | Harbor 格式容器化, 命令级交互 |
| Search 环境 (多跳搜索) | 数千实例 | 多跳检索验证, 端到端准确率评估 |

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

### 4.3 上下文管理 — Agentic 多轮上下文策略

多轮 Agent 交互中上下文随轮次线性增长，超过 100K token 后准确率显著下降 (GLM-5 实证)。需要主动管理上下文。

#### Agentic 上下文管理策略 (参考 SLIME/GLM-5)

| 策略 | 机制 | 适用场景 | 效果 |
|------|------|---------|------|
| **Keep-recent-k** | 交互历史超过阈值时，折叠早于最近 k 轮的观察 (`o_i ← "Tool result is omitted to save tokens."`)。实验中 k=5 效果最佳 | 通用 Agent 任务 | 平衡上下文长度与信息保留 |
| **Discard-all** | 重置上下文，删除全部工具调用历史 (与 DeepSeek-V3.2、Kimi K2.5 相同策略) | 轮次间独立性强的任务 | 最激进的截断 |
| **HCM (Hybrid Context Management)** | 结合 Keep-recent-k 和选择性保留 | 复杂多步任务 (如 BrowseComp) | BrowseComp 62% → 75.9% (+14%) |

#### Interleaved / Preserved Thinking (参考 GLM-5)

- **Interleaved Thinking**: 模型在每次响应和工具调用前都进行思考，提升指令遵循和生成质量
- **Preserved Thinking**: 编码 Agent 场景下，自动保留所有跨多轮对话的 thinking blocks，复用已有推理而非从头重新推导
- **Turn-level Thinking**: 支持会话内逐轮控制推理开关——轻量请求关闭以减少延迟，复杂任务开启以提升准确性

#### 配置建议

```yaml
agent:
  context_management:
    strategy: keep_recent_k     # keep_recent_k | discard_all | hcm
    k: 5                         # keep-recent-k 保留最近 k 轮
    max_context_tokens: 65536    # 触发 CM 的上下文长度阈值
    fold_message: "Tool result is omitted to save tokens."
  thinking:
    mode: interleaved            # interleaved | preserved | turn_level
    preserve_across_turns: true  # 保留跨轮次 thinking blocks
```

### 4.4 Training 层 — Megatron + GRPO

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

### 4.5 奖励层 — 混合奖励系统

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

### 4.6 数据层 — DataProto + 异步

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

#### 异步训练模式与 Off-Policy 控制

```
┌─────────────┐         ┌──────────┐         ┌─────────────┐
│  Rollout    │  ──▶    │  Buffer  │  ──▶    │  Training   │
│  Workers    │         │  (版本   │         │  Workers    │
│  (持续生成) │         │   管理)  │         │  (持续训练) │
└─────────────┘         └──────────┘         └─────────────┘
```

##### IcePop Off-Policy 控制 (参考 SLIME/GLM-5 生产实践)

异步 Agentic RL 中 Off-Policy 偏差尤为严重 (单条轨迹可跨多个策略版本)。GLM-5 的 IcePop 机制提供多层防线:

| 层次 | 机制 | 说明 |
|------|------|------|
| **IcePop pop 操作** | 重要性采样比率 ρ 超出 `[1/β, β]` 范围时**置零** | 彻底抑制极端 off-policy 样本，比 clip 更激进 |
| **Token-level Clipping** | 对 log-prob 比率应用 token 级裁剪 `[1-ε_l, 1+ε_h]` | 细粒度控制，无需跟踪历史策略检查点 |
| **版本丢弃** | 轨迹策略版本序列 `(w0,...,wk)` 中 `w'-w0 > τ` 则丢弃 | 宏观层面过滤过时数据 |
| **环境噪声过滤** | 记录失败原因，排除环境崩溃 (非模型能力) 导致的失败样本 | 防止环境不稳定污染训练信号 |
| **不完整 Group 补齐** | 噪声过滤后 group 不完整时，复制有效样本补齐或整组丢弃 | 保持 GRPO 组内归一化的统计有效性 |
| **优化器重置** | 每次权重同步后重置优化器状态 (Adam momentum 等) | 因为异步训练中优化问题本身在变化 |

```python
# IcePop pop 操作伪代码
def icepop_ratio(log_prob, old_log_prob, beta=3.0):
    rho = exp(log_prob - old_log_prob)
    # pop: 超出 [1/β, β] 范围的比率置零 (比 clip 更激进)
    mask = (rho >= 1.0 / beta) & (rho <= beta)
    return rho * mask  # 极端 off-policy 样本梯度为零

# Token-level Clipping
def token_clip(log_prob, old_log_prob, eps_l=0.2, eps_h=0.3):
    ratio = exp(log_prob - old_log_prob)
    return clip(ratio, 1 - eps_l, 1 + eps_h)

# 版本丢弃
def version_filter(trajectory, current_version, tau=3):
    oldest_version = min(trajectory.policy_versions)
    return (current_version - oldest_version) <= tau
```

### 4.7 I/O 层 — 权重同步与通信

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

### 4.8 容错 — Heartbeat 与故障恢复

#### Heartbeat 容错 (参考 SLIME/GLM-5)

大规模 Rollout 中推理 Server 故障是常见事件。GLM-5 的 Heartbeat 机制:

| 组件 | 机制 | 说明 |
|------|------|------|
| **心跳发送** | Rollout Server 定期 (每 5-10s) 发送心跳 | 编排层持续监控 Server 健康 |
| **故障检测** | 连续 N 次心跳超时 → 判定不健康 | 避免瞬时网络抖动误判 |
| **主动终止** | 不健康 Server 被主动终止 | 防止僵死 Server 占用资源 |
| **路由注销** | 从推理路由器 (SlimeRouter) 中注销 | 新请求不再路由到故障 Server |
| **自动重试** | 进行中的请求自动路由到健康 Server | 透明容错，上层无感知 |

```
正常运行:                           故障场景:
  Server_A ❤️ → Orchestrator          Server_A ❌ → Orchestrator
  Server_B ❤️ → Orchestrator            ├─ 终止 Server_A
  Server_C ❤️ → Orchestrator            ├─ 从路由器注销 A
                                        └─ Server_A 的请求 → 重试到 B/C
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

DeepSeek-V4-Flash (284B MoE, 13B 活跃参数):
  64 GPU 配置:
    TP=8 (单节点 8 GPU)
    PP=2 (2 个 Pipeline Stage)
    DP=4 (4 路数据并行)
    EP=8 (8 路专家并行)
  特性: CSA/HCA 注意力, KV Cache 减少 90%, FP4 推理
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

## 7. DeepSeek V4 启发的训练流水线优化

> 基于 DeepSeek V4 技术报告 (2026年4月) 的关键创新，对现有训练方案进行升级。

### 7.1 Specialist Training → OPD 两阶段训练范式

V4 用 On-Policy Distillation (OPD) 完全替代了传统的混合 RL 阶段。我们可借鉴此范式：

#### 阶段 1: Specialist RL Training

为不同领域分别训练专家模型：
- 数学推理专家 (GRPO, Think Max 模式)
- 代码生成专家 (GRPO, Think High 模式)
- 工具调用专家 (GRPO, 本方案主力)
- 通用对话专家 (GRPO, Non-think 模式)

每个 Specialist 独立训练，避免多任务间的奖励信号冲突。

#### 阶段 2: On-Policy Distillation (OPD)

将多个 Specialist 合并为通用模型：
- 学生模型自己生成样本 (On-Policy)
- 按教师索引排序 mini-batch，减少教师切换开销
- 全词表 KL 蒸馏 (非 Top-K logits)
- 隐藏状态缓存替代 logits 物化，节省显存

#### 配置参考

参见 `configs/opd_deepseek_v4.yaml` 配置模板。

### 7.2 Generative Reward Model (GRM)

V4 的创新：Actor 模型同时作为 GRM，联合优化评估和生成能力。

实现方案：
- 训练数据中加入 Rubric-Guided 评估样本
- Actor 在 RL 训练时同时学习生成回答和评估回答质量
- GRM 不输出标量分数，而是生成 Rubric-Guided 的评价文本
- 从评价文本中提取结构化分数用于 RL 奖励

优势：
- 消除独立 Reward Model 的资源开销
- Actor 对任务理解更深，评估更准确
- 支持复杂任务的 Rubric-Based 评估

### 7.3 多 Reasoning Effort 模式

V4 支持三种推理努力程度：

| 模式 | 系统提示注入 | 适用场景 |
|------|-----------|---------|
| Non-think | 无特殊指令 | 简单事实查询 |
| Think High | 标准推理提示 | 中等复杂度任务 |
| Think Max | "Please think step by step carefully" | 高复杂度推理 |

训练时混合使用三种模式，使模型学会根据任务复杂度调整推理深度。

### 7.4 FP4 量化推理方案

V4 使用 MXFP4 格式实现 FP4 量化：

- **存储**: FP4 格式 (4-bit 浮点)
- **计算**: 动态反量化到 FP8 进行矩阵乘法
- **精度**: FP4→FP8 反量化几乎无损
- **加速**: Rollout 推理和教师模型推理均可受益
- **依赖**: TileLang 高效 kernel 框架

配置：
```yaml
quantization:
  enable: true
  method: mxfp4
  dequant_target: fp8        # FP4→FP8 反量化
  apply_to:
    - rollout                 # Actor 推理
    - reference               # 参考模型
    - teacher                  # OPD 教师模型
```

### 7.5 可抢占容错 Rollout (Token-WAL)

V4 的 Token-WAL 机制保障大规模 Rollout 的容错性：

- **Write-Ahead Log**: 每生成 N 个 Token 持久化到 WAL
- **KV Cache 持久化**: 抢占时保存 KV Cache 状态
- **恢复**: 从 WAL + KV Cache 恢复，无需重新生成
- **长度偏差校正**: 长序列更易被抢占，需校正数据分布

实现路径：
1. 短期：使用 verl 内置断点恢复 + checkpoint
2. 中期：集成 3FS 分布式文件系统用于 KV Cache 持久化
3. 长期：完整 Token-WAL 实现

### 7.6 百万 Token 上下文 RL 训练

V4 支持百万 Token 上下文的 RL 训练，关键优化：

#### 数据格式优化

```
传统: DataProto { input_ids[B,S], attention_mask[B,S], rewards[B], ... }
                  ↑ 所有字段统一存储，S=1M 时内存爆炸

V4:   metadata { prompt_id, reward, num_turns, ... }     # 样本级，体积小
      per_token { input_ids[S], attention_mask[S], ... }  # per-token，按需加载
```

#### 共享内存数据加载器

多个 Worker 通过 shared memory 共享大序列数据，避免复制。

#### 动态 mini-batch

根据序列长度动态调整 mini-batch 大小：长序列 → 小 batch，短序列 → 大 batch。

### 7.7 沙箱升级 — 参考 DSec

V4 的 DSec 沙箱平台为 Agentic RL 提供了生产级参考：

#### 4 种执行基底

| 基底 | 用途 | 启动时间 |
|------|------|---------|
| Bare Container | 简单脚本执行 | ~100ms |
| VM-backed Container | 系统级任务 | ~1s |
| Browser Container | Web 交互 | ~2s |
| Remote Desktop | GUI 任务 | ~5s |

#### 升级路径

1. 当前：Docker 容器 (Bare Container 等价)
2. 下一步：增加 Browser Container (Headless Chrome)
3. 长期：分层存储加速镜像加载 + 轨迹日志

### 7.8 三阶段 RL 流水线 (参考 GLM-5)

GLM-5 的后训练采用渐进式 RL 流水线，各阶段对 Infra 有不同要求:

```
Reasoning RL ──→ Agentic RL ──→ General RL ──→ OPD (可选)
  (同步)          (完全异步)       (混合)         (同步蒸馏)
```

| 阶段 | 训练模式 | Infra 重点 | 算法 |
|------|---------|-----------|------|
| **1. Reasoning RL** | 完全 On-Policy, 同步 | 标准 RL Infra, 无需环境管理 | GRPO + IcePop, group_size=32 |
| **2. Agentic RL** | 完全异步, 推理-训练解耦 | PD 解耦, Heartbeat, Multi-Task Orchestrator, 10K+ 环境, Token-clip | GRPO + IcePop + Token-level Clip |
| **3. General RL** | 混合 | 多维度奖励系统 (Rule + Judge + RM) | 混合奖励 |
| **4. OPD** (可选) | 同步蒸馏 | 多教师调度, group_size=1, batch_size=1024 | 全词表 KL 蒸馏 |

**Agentic RL 阶段的 Infra 关键差异**:
- 需要 PD 解耦 (多轮 Agent 长前缀 Prefill 干扰 Decode)
- 需要 Heartbeat 容错 (1000+ 并发 Server 故障频率高)
- 需要 Multi-Task Orchestrator (异构任务: SWE/Terminal/Search)
- 需要环境噪声过滤 + 不完整 group 补齐
- 需要 Token-level Clipping 控制 off-policy 偏差 (异步训练)
- 每次权重同步后重置优化器

---

## 8. 监控与调优

### 8.1 关键指标

| 类别 | 指标 | 目标值 |
|------|------|--------|
| 训练 | Policy Loss | 持续下降 |
| 训练 | KL 散度 | 0.01-0.1 |
| 训练 | Entropy | 保持适当高 (防坍塌) |
| 训练 | Gradient Norm | < max_grad_norm |
| Rollout | 平均轮次数 | 2-5 (任务相关) |
| Rollout | 工具调用成功率 | > 90% |
| Rollout | 生成吞吐量 (tok/s) | 持续监控 |
| Rollout | **尾部时间占比 (tail_time_ratio)** | **< 15%** (参考 Seer: 47%→15%) |
| 奖励 | 平均奖励 | 持续上升 |
| 奖励 | 奖励方差 | 组内方差不为零 |
| Off-Policy | **Off-policy 丢弃率 (off_policy_drop_ratio)** | **< 20%** (过高说明异步过度) |
| MoE | 路由一致率 | > 90% |
| MoE | 专家负载均衡度 | 标准差 < 0.1 |
| 系统 | GPU 利用率 | > 70% |
| 系统 | 权重同步延迟 | < 500ms |
| 系统 | **KV Cache Pool 利用率** | **60-85%** (过低浪费, 过高抢占) |
| 系统 | **Heartbeat 超时率** | **< 1%** (> 5% 需排查 Server 稳定性) |

### 8.2 常见问题与解决

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 熵坍塌 | 所有输出趋同 | 切换 DAPO + 增大 entropy_coeff |
| 奖励 hacking | 奖励上升但质量下降 | 检查奖励函数，增加 Judge 权重 |
| 训练不稳定 | Loss 震荡 | 降低 lr，增大 clip_ratio |
| Rollout 太慢 | GPU 利用率低 | 增大批量，检查工具调用延迟 |
| **Rollout 长尾** | **tail_time_ratio > 30%** | **Divided Rollout (§4.1.4) + PD 解耦 (§4.1.1)** |
| **Off-policy 漂移** | **off_policy_drop_ratio > 30%** | **IcePop pop + Token-level Clip (§4.6)** |
| OOM | GPU 显存不足 | 减小 micro_batch，启用 LoRA |
| 路由不一致 | KL 异常 | 启用 R3 路由回放 |

---

## 9. 与其他系统的对比

| 特性 | 本方案 (verl) | ROLL | SLIME (GLM-5) | Forge | Seer | DeepSeek V4 |
|------|-------------|------|--------------|-------|------|-------------|
| 多轮 Agent | AgentLoop | AgentServer + Chunked MDP | Multi-Task Orchestrator + 1000+ 并发 | Agent 多轮 | - | DSec 沙箱 + 多基底 |
| MoE 支持 | Megatron EP | DeepSpeed + Megatron | Megatron + DeepEP (744B/256专家) | 未公开 | DP32+EP32 (1T+ MoE) | 284B MoE, CSA/HCA |
| 路由一致性 | 监控 + 可选 R3 | 无特殊处理 | 确定性 top-k (DSA 冻结 Indexer) | 无特殊处理 | - | 端到端一致 |
| Off-Policy | 版本管理 + 丢弃 | Chunked MDP IS | **IcePop + Token-clip + 版本丢弃 + 噪声过滤** | Windowed FIFO | 严格同步 On-Policy | OPD 蒸馏 |
| 长尾处理 | 超时回收 | RollPacker | APRIL + PD 解耦 + Heartbeat | Windowed FIFO | **Divided Rollout + DGDS** | Token-WAL 可抢占 |
| KV Cache | vLLM 内置 | vLLM/SGLang | SGLang Radix + DP-aware 路由 | L3 全局 Pool | **Mooncake 全局 Pool** | CSA/HCA (减少 90%) |
| 上下文管理 | 动态批处理 | Scheduler | **Keep-recent-k + HCM** | CM as Action | Mooncake 分布式 | CSA/HCA 压缩 |
| 环境管理 | Docker 沙箱 | Rock + iFlow | Heartbeat 容错 + 噪声过滤 + 10K+ 环境 | 内建 | - | DSec 4 种基底 |
| 异步训练 | 支持 | 支持 | 完全异步 (优化器重置) | 支持 | 严格同步 (反超异步 43%) | 支持 |
| 最大验证规模 | 671B MoE | 3000+ GPU | **744B MoE (256专家)** | 230B MoE | **1T+ MoE (Kimi-K2)** | 284B MoE (百万上下文) |

---

## 10. 项目文件说明

详见 `verl-agent-training/` 目录:

| 文件 | 说明 |
|------|------|
| `configs/grpo_deepseek_tool.yaml` | 主训练配置 (GRPO + DeepSeek + 工具调用) |
| `configs/ppo_deepseek_tool.yaml` | PPO 备选配置 |
| `configs/opd_deepseek_v4.yaml` | OPD 蒸馏配置 |
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
