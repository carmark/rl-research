# verl Agent Training — 工具调用 Agent RL 训练

基于 [verl](https://github.com/verl-project/verl) 框架的工具调用 Agent 强化学习训练项目。使用 GRPO 算法在 DeepSeek MoE 模型上训练能够进行多轮工具调用的 AI Agent。

## 项目结构

```
verl-agent-training/
├── configs/
│   ├── grpo_deepseek_tool.yaml   # 主配置 (GRPO + DeepSeek + 工具调用)
│   └── ppo_deepseek_tool.yaml    # PPO 备选配置
├── scripts/
│   ├── train.sh                  # 训练启动脚本
│   ├── setup_cluster.sh          # 集群环境初始化
│   └── eval.sh                   # 评估脚本
├── src/
│   ├── reward_function.py        # 混合奖励函数 (Rule + LLM-Judge)
│   ├── tool_env.py               # 工具沙箱环境
│   ├── agent_loop.py             # AgentLoop 多轮工具调用
│   ├── data_processor.py         # 数据预处理
│   └── tools/                    # 内置工具集
│       ├── base.py               # Tool 基类 + Registry
│       ├── web_search.py         # 网页搜索
│       ├── calculator.py         # 数学计算
│       ├── code_executor.py      # Python 代码执行 (沙箱)
│       └── database.py           # SQL 数据库查询
├── data/
│   ├── train_prompts.jsonl       # 训练数据 (20 条示例)
│   └── eval_prompts.jsonl        # 评估数据 (10 条示例)
└── docker/
    └── Dockerfile.sandbox        # 代码执行沙箱镜像
```

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
bash scripts/setup_cluster.sh

# 或手动安装
pip install "verl>=0.7.1" vllm megatron-core ray[default] wandb
```

### 2. 单机验证 (8 GPU, Qwen2.5-7B)

```bash
bash scripts/train.sh \
    --config configs/grpo_deepseek_tool.yaml \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend fsdp \
    --gpus 8
```

### 3. 集群训练 (64 GPU, DeepSeek MoE)

```bash
# 头节点
bash scripts/setup_cluster.sh --head --gpus-per-node 8

# 工作节点 (每个)
bash scripts/setup_cluster.sh --worker --head-address <HEAD_IP>:6379

# 启动训练
bash scripts/train.sh \
    --config configs/grpo_deepseek_tool.yaml \
    --gpus 64
```

### 4. 评估

```bash
bash scripts/eval.sh \
    --checkpoint checkpoints/step_1000 \
    --data data/eval_prompts.jsonl
```

## 核心组件

### AgentLoop — 多轮工具调用

AgentLoop 实现 LLM 与工具环境的多轮交互:

```
Turn 1: LLM 生成 → 解析工具调用 → 沙箱执行 → 返回结果
Turn 2: LLM 继续推理 → 可能再次调用工具 → ...
...
Turn N: 任务完成 / 达到最大轮次
```

工具调用格式:
```xml
<tool_call>
{"name": "calculator", "arguments": {"expression": "sqrt(2) * 100"}}
</tool_call>
```

### 混合奖励系统

```
Total Reward = 0.7 × Rule-Based + 0.3 × LLM-Judge

Rule-Based:
  - 格式奖励: 工具调用 JSON 格式正确性 (±0.1)
  - 任务奖励: 最终答案正确性 (0/1)
  - 过程奖励: 每次有效工具调用 (+0.05)
  - No-op 惩罚: 不调用工具直接回答 (×0.3)
```

### GRPO 算法

- 每个 Prompt 生成 8 个 Response (group_size=8)
- 组内相对优势估计 (无需 Critic 模型)
- PPO-style 裁剪 (clip_ratio=0.2)
- KL 散度约束 (kl_coeff=0.01)

## 配置说明

关键配置项 (`configs/grpo_deepseek_tool.yaml`):

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `model.name` | DeepSeek-V2.5 | 基座模型 |
| `rollout.mode` | server | vLLM Server 模式 |
| `rollout.agent_loop.max_turns` | 10 | 最大交互轮次 |
| `algorithm.grpo.group_size` | 8 | GRPO 组大小 |
| `algorithm.grpo.clip_ratio` | 0.2 | 裁剪比率 |
| `training.backend` | megatron | 训练引擎 |
| `training.parallel.tensor_parallel_size` | 4 | TP 并行度 |
| `reward.rule_weight` | 0.7 | Rule 奖励权重 |

## DeepSeek MoE 注意事项

- **路由一致性**: 监控训练-推理间 Router 差异率，阈值 5%
- **EP 并行**: 专家并行与数据并行配合使用
- **FP8 推理**: 推荐使用 FP8 加速 Rollout，训练保持 BF16
- **LoRA**: 671B 模型建议启用 LoRA (rank=64) 降低显存需求

## 设计文档

详细的系统架构设计见 [`../agentic-rl-design.md`](../agentic-rl-design.md)。
