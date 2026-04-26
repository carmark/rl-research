# Agentic RL 训练系统基础设施 - 快速参考对比表

> 最后更新: 2026-04-26 | 分析框架参考: [The Landscape of Agentic RL for LLMs](https://arxiv.org/abs/2509.02547) (TMLR 2026)

## 系统总览

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **组织** | 字节跳动 | 阿里巴巴 | 清华 THUDM | RadixArk | MiniMax | 月之暗面 | Gensyn | Together AI |
| **开源** | ✅ GitHub | ✅ GitHub | ✅ GitHub | ✅ GitHub | ❌ 博客 | ❌ 论文 | ✅ GitHub | ✅ GitHub |
| **版本** | v0.7.1 | 持续更新 | v0.2.3 | 持续更新 | - | - | 持续更新 | 2026.02 |
| **论文** | EuroSys 2025 | arXiv 多篇 | arXiv | arXiv 2510.11370 | HF Blog | arXiv 2511.14617 | arXiv 2509.08721 | arXiv 2602.13692 |
| **架构模式** | 单控制器+SPMD | 单控制器 | 编排层解耦 | 编排层解耦 | 三层中间件 | 集中式 | P2P 去中心化 | Program感知 |
| **控制器** | 集中式 | 集中式 | 编排层 | 编排层 | 中间件 | 集中式 | 无中心 | 工作流级 |
| **耦合度** | 共置(可解耦) | 灵活(3模式) | 解耦 | 解耦 | 完全解耦 | 动态切换 | 完全解耦 | 工作流解耦 |

## Rollout / Generation

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **推理引擎** | vLLM/SGLang/TRT-LLM | vLLM/SGLang | SGLang | SGLang(FP8) | 自研 | vLLM | vLLM/本地 | vLLM/SGLang |
| **推理模式** | Server模式(v0.7) | 动态FP8 | HTTP Server | HTTP Server | PD解耦 | Divided Rollout | 本地推理 | Program-Aware |
| **投机解码** | - | - | - | 在线SFT MTP(25%+) | Dynamic MTP | DGDS(30-44%) | - | - |
| **长尾处理** | AgentLoop多轮 | Scheduler+RollPacker | APRIL部分Rollout | APRIL+过采样 | Windowed FIFO | 分段消除气泡 | - | KV-Cache感知 |
| **上下文长度** | - | - | - | - | 200K | 长CoT(8K chunk) | - | - |
| **关键优化** | 动态批处理 | RollPacker(2.03-2.56x) | 6-7x(FP8+DeepEP) | R3路由记录 | L3 KV Cache | 分段+DGDS | P2P共享 | 防KV Thrashing |

## Training / Optimization

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **训练引擎** | FSDP/Megatron | Megatron/FSDP/DS | Megatron(mbridge) | Megatron(mbridge) | 自研(Magi) | Megatron | trl+Hivemind | 依赖集成框架 |
| **并行策略** | 5D(DP+TP+PP+CP+EP) | 自动选择 | TP+PP+DP+CP+EP | TP+PP+DP+EP | 自研并行 | DP+TP | P2P AllReduce | 依赖集成框架 |
| **核心算法** | PPO/GRPO/DAPO/DPO | PPO/GRPO/DAPO | PPO/GRPO/DAPO | GRPO+TIS/MIS | CISPO | PPO/GRPO | SAPO(GRPO改进) | 依赖集成框架 |
| **训练模式** | 同步+异步 | 同步+异步(Flash) | 同步+异步(双边IS) | 同步+异步 | 混合域统一 | 严格同步On-Policy | 去中心化异步 | 依赖集成框架 |
| **量化支持** | 实验性FP8 | FP8 Rollout | - | FP8 E2E/INT4 QAT | - | - | - | - |
| **LoRA** | ✅ | ✅ | - | - | - | - | - | - |

## Data Processing

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **数据协议** | DataProto→TensorDict | 内部协议 | HTTP请求/响应 | R3 Mask+张量 | OpenAI API | 分段序列 | Gossip消息 | Program元数据 |
| **缓冲区** | TransferQueue | Resource Pool | Data Buffer+Ray Store | Data Buffer | Data Pool(分布式) | 段缓冲 | 本地存储 | Program状态 |
| **数据分发** | Dispatch.DP_COMPUTE | AutoDeviceMapping | SlimeRouter负载均衡 | 路由回放桥 | Gateway Server | DGDS动态分配 | P2P广播 | Program-Aware |
| **序列优化** | 零拷贝·RDMA | Offload/Reload | Sequence Packing | FP8精度一致 | Prefix Tree(~40x) | 分段拼接重组 | - | KV Cache保持 |

## I/O & Communication

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **调度框架** | Ray | Ray | Ray | Ray | 自研中间件 | 自研 | Hivemind | Program-Aware |
| **集合通信** | NCCL | NCCL | NCCL | NCCL | 自研 | NCCL | P2P AllReduce | 依赖后端 |
| **推理通信** | gRPC/HTTP | gRPC/HTTP | HTTP(SGLang) | HTTP(SGLang) | OpenAI API | 内部RPC | libp2p | 依赖后端 |
| **权重同步** | 3D-HybridEngine/NIXL | ROLLMUX挂起/恢复 | CUDA IPC/分布式 | CUDA IPC零拷贝(50%↓) | L3 KV Cache Pool | Mooncake Checkpoint | Gossip广播 | - |
| **同步延迟** | <300ms(NCCL) | 7.87-8.33x(vs verl) | BF16~48s/FP8~100s | 50%降低 | - | - | 高(互联网) | - |

## 性能关键数据

| 系统 | 核心指标 | 数值 | 条件 |
|------|---------|------|------|
| **verl** | vs DeepSpeed-Chat | 3.67x+ | 原始基准 |
| **verl** | 异步训练加速 | 2.35-2.67x | 128 GPU · Qwen2.5-7B |
| **verl** | 最大模型 | 671B MoE | DeepSeek |
| **ROLL** | ROLLMUX 成本效率 | 1.84x | vs 标准解耦 |
| **ROLL** | RollPacker 加速 | 2.03-2.56x | vs verl |
| **ROLL** | ROLLART 加速 | 1.35-2.05x | 3000+ GPU 验证 |
| **SLIME** | APRIL 端到端加速 | 40% | 吞吐量提升 |
| **SLIME** | FP8+DeepEP 推理 | 6-7x | GLM4.5-355B |
| **SLIME** | 最大模型 | 355B MoE | 64×H100 |
| **MILES** | 投机解码加速 | 25%+ | 在线SFT MTP |
| **MILES** | 权重同步降低 | 50% | vs HTTP/RPC |
| **MILES** | 单机模型规模 | 1TB | INT4 QAT · H200 |
| **Forge** | Prefix Tree 加速 | ~40x | 训练前向传播 |
| **Forge** | CISPO vs DAPO | 2x | Qwen2.5-32B |
| **Forge** | 模型规模 | 230B | M2.5(10B active) |
| **Seer** | Rollout 吞吐量 | +74-97% | 32×8 H800 |
| **Seer** | 长尾延迟降低 | -75-93% | 生产工作负载 |
| **Seer** | DGDS 单组件贡献 | +30-44% | 最大单组件 |
| **Seer** | Divided Rollout 贡献 | +27-35% | 基础组件 |
| **ThunderAgent** | RL Rollout 吞吐量 | 1.8-3.9x | vs vLLM+SGLang Gateway |
| **ThunderAgent** | 推理服务吞吐量 | 1.5-3.6x | vs vLLM/Continuum |
| **ThunderAgent** | KV Thrashing 优化 | 7.1x→近零 | 延迟膨胀消除 |
| **rl-swarm** | 累计奖励提升 | 94% | 8×Qwen2.5-0.5B |

## Agentic RL 维度 (参考综述框架)

### RL 粒度与 MDP 建模

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **MDP 类型** | 单步→POMDP | 单步→POMDP | 单步MDP | 单步MDP | POMDP原生 | 单步MDP | POMDP | 依赖集成 |
| **RL 粒度** | Token/Trajectory | **Chunk级**灵活 | Token/Traj | Token/Traj | Turn级 | Token/Traj | Trajectory | 依赖集成 |
| **多轮交互** | AgentLoop(v0.7) | 4种异步+AgentServer | 多轮模式(更优) | MrlX多智能体 | 原生多轮 | - | 多轮代码 | Program级多轮 |
| **环境交互** | vLLM/SGLang Server | 环境级异步+Rock | HTTP+sgl-router | HTTP API | Gateway+环境 | 内部RPC | 本地沙箱 | Program感知 |

### 奖励建模方式

| 维度 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **奖励类型** | ORM+Rule+Model | ORM+Rule+Model+**Dense** | ORM+Rule+Verifier | ORM+Rule | Model+Critique | ORM+Rule+**时间** | Model-Based | 依赖集成 |
| **粒度** | Trajectory | Trajectory/Step/**Chunk** | Trajectory | Trajectory | Turn级 | Traj+**Reward-to-Go** | Trajectory | 依赖集成 |
| **RLVR支持** | ✅ | ✅ | ✅ | ✅ | 部分 | ✅ | 部分 | 依赖集成 |
| **Critic模型** | ✅ PPO | ✅ | ✅ PPO | ❌ GRPO | ❌ CISPO | ✅ PPO | ❌ SAPO | 依赖集成 |

### Agentic 能力支持度

| 能力 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **Planning** | ◐ AgentLoop | ◐ 异步流水线 | ○ | ○ | ● Gateway编排 | ◐ Divided | ◐ GameManager | ◐ Program DAG |
| **Tool Use** | ◐ Server API | ● 环境级+AgentServer | ◐ HTTP+sgl-router | ◐ HTTP | ● 数千种工具 | ◐ | ◐ 代码执行 | ● Docker容器 |
| **Memory** | ○ | ○ | ◐ RadixTrie | ◐ RadixTrie | ● KV Cache Pool | ● Mooncake | ○ | ● KV Cache感知 |
| **Reasoning** | ● GRPO/DAPO | ● GRPO/DAPO+Chunked | ● GRPO/DAPO | ● R3+GRPO | ● CISPO | ● DGDS | ◐ SAPO | 依赖后端 |
| **Self-Improve** | ◐ 异步迭代 | ◐ Flash | ◐ 异步 | ◐ 在线SFT | ● 混合域 | ○ | ◐ Swarm传播 | ○ |
| **Multi-Agent** | ○ | ○ | ◐ OpenClaw-RL | ● MrlX | ● 多Agent | ○ | ● P2P Swarm | ● Multi-Program |
| **Env Mgmt** | ○ | ● Rock+清理+验证 | ◐ 解耦设计 | ○ | ● Gateway隔离 | ○ | ◐ 本地沙箱 | ● Program隔离 |

> ● 完整支持 | ◐ 部分支持 | ○ 不支持/未明确

### Agentic 特有挑战 (新增)

| 挑战 | verl | ROLL | SLIME | MILES | Forge | Seer | rl-swarm | ThunderAgent |
|------|------|------|-------|-------|-------|------|----------|-------------|
| **Off-Policy控制** | 异步+版本管理 | Chunked MDP IS | 双边IS修正 | TIS/MIS | 滑动窗口 | 严格同步 | Gossip异步 | 依赖集成 |
| **环境泄露防护** | - | ✅严格清理+隔离 | - | - | Gateway隔离 | - | - | Program隔离 |
| **数据质量** | - | LLM-as-Judge+No-op | - | - | Evaluator | - | 交叉评估 | - |
| **上下文管理** | 动态批处理 | Scheduler | RadixTrie | RadixTrie | **CM as Action** | Mooncake分布式 | 本地 | **防KV Thrashing** |
| **Dense Reward** | - | ✅过程+时间+R2G | - | - | Critique | ✅时间+R2G | - | - |
| **KV Cache** | vLLM/SGLang内置 | vLLM/SGLang内置 | SGLang Radix | SGLang Radix | L3全局Pool(DFS) | Mooncake(DRAM+SSD) | 本地 | 防驱逐(7.1x优化) |

### RL 算法族谱

| 算法族 | 特点 | 系统覆盖 |
|--------|------|---------|
| **PPO (需Critic)** | 裁剪策略比率, KL惩罚 | verl, ROLL, SLIME, Seer |
| **GRPO (无Critic)** | 组内相对优势, 消除Critic | verl, ROLL, SLIME, MILES, Seer |
| **DAPO** | 解耦Clip+动态采样, 防熵坍塌 | verl, ROLL, SLIME |
| **CISPO** | 裁剪IS权重, 全Token梯度 | Forge |
| **SAPO** | 共享Rollout文本(非梯度) | rl-swarm |
| **Chunked MDP** | Chunk级IS, 匹配Agent交互 | ROLL |
| **DPO/SimPO** | 无RL循环, 偏好分类 | verl (辅助) |

## 创新点速查

| 系统 | 核心创新 | 解决的问题 |
|------|---------|-----------|
| **verl** | HybridFlow 混合控制器 | 灵活性与性能的统一 |
| **verl** | 3D-HybridEngine | 训练↔推理零冗余权重切换 |
| **verl** | TransferQueue | 控制流与数据流解耦 |
| **ROLL** | ROLLMUX 阶段级复用 | GPU 资源利用率最大化 |
| **ROLL** | ROLLART Serverless | 无状态组件弹性伸缩 |
| **ROLL** | Chunked MDP + AgentServer | Agentic 环境管理与奖励建模 |
| **ROLL** | LLM-as-Judge + No-op 验证 | 数据质量保障与伪阳性过滤 |
| **SLIME** | APRIL 主动部分 Rollout | 长尾生成延迟优化 |
| **SLIME** | sgl-router Agent-RL 解耦 | Agent 框架与 RL 框架分离 |
| **SLIME** | 双边 IS 采样修正 | 异步训练 Off-Policy 偏差控制 |
| **MILES** | R3 路由回放 | MoE 训练-推理一致性 |
| **MILES** | 端到端 FP8 | 精度一致的量化加速 |
| **MILES** | 在线 SFT 投机解码 | 草稿模型与训练同步 |
| **Forge** | CISPO | 全 Token 梯度参与 |
| **Forge** | Prefix Tree Merging | 共享前缀训练加速 (~40x) |
| **Forge** | CM as Agent Action | 上下文管理建模为Agent动作 |
| **Forge** | 滑动窗口 (Windowed FIFO) | Off-Policy 异步训练平衡 |
| **Forge** | Dynamic MTP + PD 解耦 | 推理加速 + MoE dispatch 隔离 |
| **Seer** | Divided Rollout | 长 CoT 场景气泡消除 (+27-35%) |
| **Seer** | DGDS 投机解码 | 无需独立草稿模型 (+30-44%) |
| **Seer** | Mooncake KVCache | 分布式两层 KV 缓存 |
| **Seer** | 任务时间奖励 + Reward-to-Go | Agentic 场景 Dense Reward |
| **ThunderAgent** | Program 抽象 | 工作流级调度单元 |
| **ThunderAgent** | KV-Cache 感知调度 | 消除 KV-Cache Thrashing (7.1x) |
| **ThunderAgent** | 异步环境准备 | 工具调用期间预准备推理环境 |
| **rl-swarm** | SAPO | 去中心化 Rollout 共享 |
| **rl-swarm** | P2P 训练 | 消费级 GPU 民主化 |

## 架构模式光谱

```
集中式 ←——————————————————————————————————————————→ 去中心化

verl    Seer    ROLL    SLIME   MILES   Forge  ThunderAgent  rl-swarm
 │       │       │       │       │       │        │           │
单控制器 集中式  灵活    编排层  编排层  中间件   Program感知  P2P
共置    动态切换 3模式   解耦    模块化  完全解耦 工作流级     无中心
```

## 技术栈依赖图

```
推理引擎:  vLLM ←── verl, ROLL, Seer, rl-swarm, ThunderAgent
          SGLang ←── verl, ROLL, SLIME, MILES, ThunderAgent
          自研 ←── Forge, Seer

训练引擎:  Megatron-LM ←── verl, ROLL, SLIME, MILES, Seer
          FSDP ←── verl, ROLL, MILES
          DeepSpeed ←── ROLL
          自研 ←── Forge
          trl/HF ←── rl-swarm

调度框架:  Ray ←── verl, ROLL, SLIME, MILES
          自研 ←── Forge, Seer
          Program-Aware ←── ThunderAgent
          Hivemind ←── rl-swarm

通信:     NCCL ←── verl, ROLL, SLIME, MILES, Seer
          NIXL ←── verl
          P2P/Gossip ←── rl-swarm
```

## Infra 演进时间线

```
2024 (RLHF 时代)          2025 (RL Scaling 时代)           2025-2026 (Agentic RL 时代)
─────────────────          ──────────────────────           ───────────────────────────
TRL/DeepSpeed-Chat         verl/OpenRLHF                   Forge异步Data Pool
短序列·同步批处理          异构调度·长CoT                  SLIME双边IS修正
单步MDP·人类偏好           671B MoE·GRPO族                 ROLL Chunked MDP+AgentServer
                           vLLM/SGLang标准化               Seer Divided Rollout
                                                           ThunderAgent Program抽象
                                                           POMDP·多轮交互·变长轨迹
                                                           CM as Action·环境泄露防护
```

---

> 详细分析请参阅 [完整报告](report.md) | 动态架构图请打开 [diagrams/index.html](diagrams/index.html)
>
> 参考来源: [知乎 - Agentic RL Infra 重构](https://zhuanlan.zhihu.com/p/2022786148087464077) | [attack204 综述](https://qingkeai.online/archives/26-Agentic-RL-Infra) | [ROLL 团队实践](https://zhuanlan.zhihu.com/p/2006389553703982040)
