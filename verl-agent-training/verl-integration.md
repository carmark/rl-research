# verl 集成指南 — 自定义 AgentLoop + Reward + Tool 的构建方式

> 本文档说明如何将自定义的 AgentLoop、Reward Function、Tool 接入 verl 框架进行 Agentic RL 训练。

---

## 1. 整体架构：三个扩展点

verl 提供三个核心扩展接口。你写的自定义代码通过这三个接口注入框架：

```
verl 训练循环 (PPOTrainer)
    │
    ├── ① AgentLoopBase.run()         ← 你控制 rollout 逻辑
    │       │
    │       ├── AsyncLLMServerManager.generate()   (verl 提供)
    │       └── BaseTool.execute()     ← ② 你实现工具执行
    │
    └── ③ compute_score()             ← 你定义奖励计算
```

### 数据流全景

```
PPOTrainer
  │
  ▼
AgentLoopManager.generate_sequences(batch: DataProto)
  │
  ├─ 唤醒 vLLM/SGLang Server (同步权重)
  │
  ├─ 分发 batch → AgentLoopWorker (Ray Actor, 并发)
  │     │
  │     └─ 对每个 sample 启动 AgentLoopBase.run() 协程
  │           │
  │           ├─ Turn 1: LLM generate → parse tool calls → Tool.execute()
  │           ├─ Turn 2: 拼接 tool response → LLM generate → ...
  │           └─ Turn N: 返回 AgentLoopOutput
  │                 │
  │                 ├─ prompt_ids:      [token, ...]
  │                 ├─ response_ids:    [token, ...]  (LLM + tool response tokens)
  │                 ├─ response_mask:   [1,1,..,0,0,..,1,1,..]
  │                 │                    ↑ LLM tokens    ↑ tool tokens (masked)
  │                 ├─ reward_score:    float (可选, 在 AgentLoop 中直接算)
  │                 └─ extra_fields:    {tool_rewards: [...], num_turns: N}
  │
  ├─ 后处理: pad 到固定长度 → 组装 DataProto batch
  │
  ├─ [如果没有 streaming reward]
  │     RewardLoopManager.compute_rm_score()
  │       └─ 调用你的 compute_score() 函数
  │
  ├─ 计算 ref_log_probs (参考策略)
  ├─ 计算 advantage (GRPO 组内归一化)
  └─ 更新 actor (PPO clip + 梯度)
       └─ 权重同步回 vLLM Server
```

---

## 2. 扩展点 ①：AgentLoop (Rollout 逻辑)

### 接口定义

```python
# verl/experimental/agent_loop/agent_loop.py

class AgentLoopBase(ABC):
    def __init__(
        self,
        trainer_config,          # 全局训练配置
        server_manager,          # AsyncLLMServerManager (调用 LLM)
        tokenizer,               # 分词器
        processor,               # 多模态处理器 (可选)
        dataset_cls,             # 数据集类
        data_config,             # 数据配置
        **kwargs,
    ): ...

    @abstractmethod
    async def run(self, sampling_params: dict, **kwargs) -> AgentLoopOutput:
        """
        kwargs 包含 dataset 中的字段:
            raw_prompt, extra_info, agent_name 等
        """
        raise NotImplementedError
```

### AgentLoopOutput — run() 必须返回的结构

```python
class AgentLoopOutput(BaseModel):
    prompt_ids: list[int]              # Prompt token IDs
    response_ids: list[int]            # 完整 response token IDs (LLM + tool tokens)
    response_mask: list[int]           # 1=LLM 生成, 0=tool response / padding
    response_logprobs: list[float]     # 每个 token 的 log-prob (可选)
    reward_score: float | None         # 在 AgentLoop 中直接算的奖励 (可选)
    num_turns: int                     # 交互轮次
    metrics: AgentLoopMetrics          # 吞吐量等指标
    extra_fields: dict[str, Any]       # 自定义数据 (tool_rewards, turn_scores 等)
```

**核心概念 — response_mask**:

```
多轮交互的 response_ids:
  [LLM tokens] [tool response tokens] [LLM tokens] [padding]

response_mask:
  [1, 1, .., 1] [0, 0, ...., 0, 0]   [1, 1, .., 1] [0, 0, .., 0]

只有 mask=1 的 token 参与策略梯度计算。
tool response tokens 是环境给的，不是模型生成的，所以 mask=0。
```

### 两种实现方式

**方式 A — 使用 verl 内置 ToolAgentLoop (推荐)**

verl 已内置 `ToolAgentLoop`，实现了完整的状态机:
```
PENDING → GENERATING → (PROCESSING_TOOLS → GENERATING)* → TERMINATED
```

你只需要配置即可，无需写代码:

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      max_assistant_turns: 10
      tool_config_path: configs/tool_config.yaml
    agent:
      default_agent_loop: tool_agent   # 内置的 ToolAgentLoop
```

**方式 B — 自定义 AgentLoop (高级)**

当内置 ToolAgentLoop 不满足需求时 (如需要超时处理、自定义终止条件):

```python
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

@register("tool_call_agent")
class ToolCallAgentLoop(ToolAgentLoop):
    """继承 ToolAgentLoop 并添加自定义逻辑。"""

    async def run(self, sampling_params, **kwargs):
        # 自定义: 添加超时控制
        import asyncio
        try:
            output = await asyncio.wait_for(
                super().run(sampling_params, **kwargs),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            output = self._build_timeout_output(kwargs)
        return output
```

然后在 config 中引用:
```yaml
actor_rollout_ref.rollout.agent.default_agent_loop: tool_call_agent
```

---

## 3. 扩展点 ②：Tool (工具实现)

### 接口定义

```python
# verl/tools/base_tool.py

class BaseTool:
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        self.config = config
        self.tool_schema = tool_schema
        self.name = tool_schema.function.name

    async def create(self, instance_id=None, **kwargs):
        """为每条轨迹创建一个工具实例。"""
        return instance_id, ToolResponse(text="Ready")

    async def execute(self, instance_id: str, parameters: dict, **kwargs):
        """执行工具。返回 (response, step_reward, metrics)。"""
        return ToolResponse(text="result"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """基于工具状态计算最终奖励 (可选)。"""
        return 0.0

    async def release(self, instance_id: str, **kwargs):
        """清理工具实例。"""
        pass
```

### 生命周期

```
create()    ← 每个 episode 开始时调用 (初始化沙箱、DB 连接等)
    │
    ├── execute()  ← 每次工具调用时调用
    │       返回 (ToolResponse, step_reward, metrics)
    │       step_reward 收集到 extra_fields["tool_rewards"]
    ├── execute()
    │   ...
    │
    ├── calc_reward()  ← episode 结束时调用 (汇总奖励)
    │
    └── release()  ← 清理资源
```

### 注册方式 — YAML 配置文件

```yaml
# configs/tool_config.yaml
tools:
  - class_name: "src.tools.verl_tools.CalculatorTool"
    config:
      type: native
    tool_schema:
      type: "function"
      function:
        name: "calculator"
        description: "Evaluate a mathematical expression"
        parameters:
          type: "object"
          properties:
            expression:
              type: "string"
              description: "Math expression, e.g. 'sqrt(2) + pi'"
          required: ["expression"]
```

在训练配置中引用:
```yaml
actor_rollout_ref.rollout.multi_turn.tool_config_path: configs/tool_config.yaml
```

### 关键设计点

1. **step_reward**: `execute()` 返回的第二个值是步级奖励，verl 自动收集到 `extra_fields["tool_rewards"]`，你的 `compute_score()` 可以利用这些信息
2. **instance_id**: 每条轨迹有独立的 instance_id，实现环境隔离
3. **异步**: 所有方法都是 `async`，支持并发工具执行

---

## 4. 扩展点 ③：Reward Function (奖励函数)

### 接口定义 — compute_score()

```python
def compute_score(
    data_source: str,           # 数据集名称
    solution_str: str,          # 模型的完整回答文本
    ground_truth: str,          # 数据集中的标准答案
    extra_info: dict = None,    # 元数据
    **kwargs,
) -> float | dict:
    """
    返回:
      - float: 直接作为奖励分数
      - dict: 必须包含 "score" 键, 其他键用于 logging
    """
```

### extra_info 中包含什么

当使用 ToolAgentLoop 时，`extra_info` 自动包含:

```python
extra_info = {
    "num_turns": 5,                    # 交互轮次
    "tool_rewards": [0.05, 0.05, 0.0], # 每步工具执行的 step_reward
    "turn_scores": [...],              # 每轮分数
    "rollout_reward_scores": [...],    # rollout 阶段的奖励
    # ... 数据集中的原始 extra_info 字段
}
```

### 注册方式

```yaml
# 训练配置中
reward:
  custom_reward_function:
    path: src/reward_function.py       # Python 文件路径
    name: compute_score                # 函数名
```

### 我们的实现

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # 1. 任务完成度 (答案匹配)
    task_correct = ground_truth.lower() in solution_str.lower()
    task_score = 1.0 if task_correct else 0.0

    # 2. 格式质量 (tool call JSON 格式)
    format_score = check_format(solution_str)

    # 3. 工具使用奖励 (来自 BaseTool.execute 的 step_reward 累加)
    tool_rewards = extra_info.get("tool_rewards", [])
    tool_score = sum(tool_rewards)

    # 4. No-op 惩罚
    is_noop = not bool(tool_rewards) and "<tool_call>" not in solution_str

    total = 0.7 * (task_score + format_score) + 0.3 * tool_score
    if is_noop:
        total *= 0.3

    return {"score": total, "task_correct": task_correct, "is_noop": is_noop}
```

---

## 5. 完整配置示例

### 目录结构

```
verl-agent-training/
├── configs/
│   ├── grpo_deepseek_tool.yaml   # verl Hydra 训练配置
│   ├── ppo_deepseek_tool.yaml    # PPO 备选
│   └── tool_config.yaml          # 工具注册 (BaseTool YAML)
├── src/
│   ├── reward_function.py        # compute_score() 函数
│   ├── agent_loop.py             # 自定义 AgentLoop (可选)
│   └── tools/
│       └── verl_tools.py         # BaseTool 子类实现
└── data/
    └── train_prompts.jsonl       # 训练数据
```

### 启动命令

```bash
# 单机验证 (8 GPU, Qwen2.5-7B)
python3 -m verl.trainer.main_ppo \
    --config-path=configs \
    --config-name=grpo_deepseek_tool \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.strategy=fsdp \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8

# 集群训练 (64 GPU, DeepSeek MoE)
python3 -m verl.trainer.main_ppo \
    --config-path=configs \
    --config-name=grpo_deepseek_tool \
    trainer.nnodes=8 \
    trainer.n_gpus_per_node=8
```

---

## 6. 常见问题

### Q: response_mask 怎么自动生成？

A: verl 的 ToolAgentLoop 自动处理。LLM 生成的 token mask=1，tool response 拼接的 token mask=0。你不需要手动设置。

### Q: compute_score 是同步还是异步调用？

A: 都支持。如果你的奖励函数需要调用外部 API (如 LLM-as-Judge)，定义为 `async def compute_score(...)` 即可。

### Q: 怎么 debug AgentLoop？

A: 使用 `StandaloneAgentLoop` (在 `src/agent_loop.py` 中)，不依赖 verl，可以单独测试工具调用逻辑。

### Q: 工具的 step_reward 和 compute_score 的关系？

A: step_reward 是过程奖励 (每次工具调用)，compute_score 是最终奖励 (整条轨迹)。两者互补。verl 会把 step_reward 收集到 `extra_info["tool_rewards"]`，你的 compute_score 可以利用它们。

### Q: 如何使用自定义 AgentLoop 替代内置的？

A: 在训练脚本中 import 并注册:
```python
from src.agent_loop import register_tool_call_agent
register_tool_call_agent()
```
然后配置 `default_agent_loop: tool_call_agent`。
