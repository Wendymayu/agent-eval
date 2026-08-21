# agenteval — 基于 deepeval 的精简 Agent 评估框架（设计文档）

- **日期**: 2026-08-21
- **状态**: 设计已批准，待写实现计划
- **来源**: 抽取自 deepeval v4.1.8（`D:\code\opensource\github\deepeval`）
- **目标仓库**: `D:\code\opensource\github\agent-eval`

## 1. 目标与非目标

### 目标
- 构建一个精简的 LLM/Agent 评估框架，**只保留 deepeval 的核心四件**：TestCase、Metric、Model（LLM 调用）、evaluate（编排）。
- 起步阶段以**单轮 LLM 评估**为主（input→output，LLM-as-judge 对比 expected），端到端可用。
- 架构上**预留 Agent 扩展位**（多轮、工具调用、trace），数据模型不需返工即可向后扩展。
- 与 deepeval 的关系：**抽取拷贝核心代码（fork/vendor）**，作为独立项目维护，剥离可选装饰。

### 非目标（v1 不做）
- 不接 Confident AI 云上传、不上传遥测（posthog）。
- 不做 rich 控制台报告、不内置本地 sqlite 持久化、不做结果缓存、不做超时重试。
- 不做 pytest 插件 / `assert_test` 的完整集成（v1 以程序化 `evaluate()` 为主）。
- 不移植 benchmarks（GSM8K/HumanEval 等）、不移植 synthesizer/dataset 生成器、不移植 CLI。
- v1 不接 conversational 分支的 metric（数据模型到位但不接线）。

## 2. 可行性结论

可行。四个核心子系统的最小依赖闭包清晰且小：

- **TestCase**：无共享基类，`LLMTestCase` / `ConversationalTestCase` 直接继承 `pydantic.BaseModel`；`make_model_config` 等价于 `ConfigDict(extra="ignore")`。唯一硬外部依赖是 pydantic。`LLMTestCase` 已自带 `tools_called`/`expected_tools`，`ConversationalTestCase`+`Turn` 已存在 —— 单轮起步 + Agent 扩展天然契合。
- **Metric**：基类 `BaseMetric`（`deepeval/metrics/base_metric.py`）；"结果"是写回实例属性（`measure()` 设 `self.score/reason/success` 并返回 `self.score`），无单独结果对象。LLM-as-judge 走 pydantic schema + `generate_with_schema`。
- **Model**：`DeepEvalBaseLLM`（`deepeval/models/base_model.py:46`）是纯 `ABC`（非 pydantic）。最小接口：`load_model` + `generate(prompt, schema=None) -> (output, cost)` + `a_generate` + `get_model_name`。结构化输出优先 OpenAI 原生 `beta.chat.completions.parse(response_format=schema)`，降级 JSON 模式，再降级 prompt+`trim_and_load_json`+`schema.model_validate`。返回契约 `Tuple[Union[str, BaseModel], float]`（第二项为美元成本 `EvaluationCost`）。
- **evaluate**：`evaluate(test_cases, metrics)` 返回 `EvaluationResult`；evaluate 上**无 model 参数**，评判模型挂每个 metric 上。并发为 asyncio 两级并行（case 间并行 + case 内 metric 并行，metric 按 case 深拷贝）。`assert_test` 是单 case 版 + 失败抛 `AssertionError`，共用同一执行引擎。

agent 已确认：test_run 持久化、`wrap_up_test_run`、Confident 云、posthog 遥测、`EvaluationConsoleReport`、缓存、超时、pytest 钩子**全部为可选装饰**，移除后 `EvaluationResult.test_results` 的分数计算不受影响。`chromadb/langchain/tiktoken/torch/transformers/nltk/detoxify/ragas` 等重依赖现代 Metric 核心**根本不 import**，可全部留下。

## 3. 包结构

沿用 deepeval 已验证的布局，简化后重新打包为 `agenteval`：

```
agenteval/
├── __init__.py          # 公开 API: evaluate, LLMTestCase, BaseMetric, GEval, DeepEvalBaseLLM, OpenAIModel, AgentEvalError …
├── errors.py            # AgentEvalError 基类异常
├── test_case/
│   ├── __init__.py
│   ├── llm_test_case.py         # LLMTestCase, ToolCall, ToolCallType, SingleTurnParams, RetrievedContextData, MLLMImage, _MLLM_IMAGE_REGISTRY
│   └── conversational_test_case.py  # ConversationalTestCase, Turn, MultiTurnParams  ← Agent 扩展位
├── metrics/
│   ├── __init__.py
│   ├── base_metric.py          # BaseMetric (BaseConversationalMetric 留扩展位)  ← 外科: 去 tracing
│   ├── utils.py                # initialize_model, generate_with_schema_and_extract, check_llm_test_case_params  ← 外科: lazy import
│   ├── g_eval/                 # GEval（通用 LLM-judge，主力 metric）+ schema.py + template
│   └── exact_match/            # ExactMatchMetric（确定性范例 metric）
├── models/
│   ├── __init__.py
│   ├── base_model.py           # DeepEvalBaseLLM, DeepEvalModelData  ← 外科: 去 tracing
│   ├── utils.py                # EvaluationCost, parse_model_name, require_secret_api_key, require_costs
│   └── llms/
│       ├── __init__.py
│       ├── openai_model.py     # OpenAIModel（默认 provider）
│       └── utils.py            # trim_and_load_json
├── evaluate/
│   ├── __init__.py
│   ├── evaluate.py             # evaluate()  ← 外科: 去报告/持久化/云
│   ├── types.py                # EvaluationResult, TestResult
│   ├── execute/
│   │   ├── __init__.py
│   │   └── e2e.py              # execute_test_cases / a_execute_test_cases（sync+async）
│   └── utils.py                # create_metric_data, create_test_result
└── templates/                  # jinja2 模板包（GEval 提示词）；移植 resolver.py + template_class.py(make_template_class) + metrics/templates.json[+fragments]
    ├── metrics/templates.json
    ├── resolver.py              # TemplateRegistry + resolve_template（jinja2 StrictUndefined 渲染）
    └── template_class.py        # make_template_class：合成 <Metric>Template 静态方法类
```

### 命名约定（决策点①：已定）
- 内部类名**保持 deepeval 原名**（`DeepEvalBaseLLM`、`BaseMetric`、`LLMTestCase`、`GEval`…），忠实 fork、减少跨引用改坏风险。
- 仅包路径与基类异常改名为 `agenteval` / `AgentEvalError`。用户经 `from agenteval import ...` 访问。
- **为何只改异常名**：`DeepEvalError` 是项目根命名空间的入口异常，改成 `AgentEvalError` 是新项目身份所需；而评估类（`DeepEvalBaseLLM` 等）保持原名以忠实 fork、减少跨引用风险。后续若要彻底改名为 `AgentEval*`，作为独立重构任务再做。

## 4. 外科剥离清单

保留评分逻辑，砍掉可选装饰。每项均经 agent 确认不影响分数计算。

| # | 改动 | 位置 | 做法 |
|---|---|---|---|
| 1 | **去 tracing** | `models/base_model.py`、`metrics/base_metric.py` 的 `__init_subclass__` | 改 no-op，断开 `deepeval.tracing` 依赖。否则定义一个 metric/model 子类就触发 OTel 导入。 |
| 2 | **lazy provider import** | `metrics/utils.py` 的 `initialize_model` | 删除顶层 `from ..models import (OpenAIModel, AnthropicModel, AzureOpenAIModel, …)`；改为按需 import；无 model 时默认 `OpenAIModel`。这是抽取的关键去耦点。 |
| 3 | **pydantic v2 only** | `make_model_config` | 直接 `ConfigDict(extra="ignore")`，删 v1 分支（deepeval 原 `utils.py:41` 双版本逻辑不再需要）。 |
| 4 | **evaluate 瘦身** | `evaluate/evaluate.py` | 保留：校验输入（至少一个 metric 有 threshold）→ 分发到引擎 → 组装返回 `EvaluationResult`。砍：`EvaluationConsoleReport`、`wrap_up_test_run`、`post_test_run`/Confident 云、`metric_collection` 分支、缓存（`cache_config`）、超时重试。 |
| 5 | **去遥测** | posthog `capture_evaluation_run` | 删除或改 no-op contextmanager；不引入 `posthog` 依赖。 |

## 5. 数据流

### 正常路径（async 默认）
```
evaluate(test_cases, metrics)
  → 校验（至少一个 metric 有 threshold）
  → a_execute_test_cases(test_cases, metrics, async_config)   [Semaphore(max_concurrent=20)]
      → 每个 test_case 并发：
          - 深拷贝 metrics（copy_metrics，避免并行共享可变状态）
          - measure_metrics_with_indicator: asyncio.gather → 每个 metric.a_measure(test_case)
              └─ metric.model.generate(prompt, schema=SchemaCls) → (output, cost)
                 metric 设 self.score / self.reason / self.success / self.evaluation_cost / self.error
          - create_metric_data(metric) → MetricData
          - 聚合进 LLMApiTestCase（case 成功 = 所有非 flaky metric 成功）
      → create_test_result(api_test_case) → TestResult
  → return EvaluationResult(test_results=[TestResult, …], confident_link=None, test_run_id=None)
```

### sync 路径
顺序执行：逐 case、逐 metric 调 `metric.measure(test_case)`，同样经 `create_metric_data` → `create_test_result`。

### 关键契约（沿用 deepeval，不改）
- evaluate 无 model 参数；评判模型挂每个 metric（`metric.model`）。
- metric 结果为实例状态（`self.score` 等），非返回对象。
- `generate()` 返回 `Tuple[Union[str, BaseModel], float]`（output, USD cost）。

## 6. 错误处理

- **metric 出错不炸整批**：单个 metric 抛异常被捕获 → 设 `metric.error`、`success=False`，评估继续。对批量评估关键。保留精简 `ErrorConfig`（continue/raise 两档；默认 continue）。
- 缺 API key → `require_secret_api_key` 抛 `AgentEvalError` + 环境变量提示。
- 缺可选 provider SDK → `require_dependency` 抛 `AgentEvalError` + 安装提示（`pip install <pkg>`）。
- LLM 返回非 JSON → `trim_and_load_json` 抛 `AgentEvalError`（保留 deepeval 行为：找首个 `{` 到末个 `}`，strip 尾逗号，`json.loads`）。
- 基础异常统一为 `AgentEvalError`（取代 `DeepEvalError`）。

## 7. 测试策略

- **MockModel**：移植 deepeval 的 mock `DeepEvalBaseLLM`（返回预设 `(output, cost)`），CI 不打真实 API。确定性测试关键。
- 移植 deepeval 自带相关单测：TestCase 序列化、GEval 用 MockModel 打分、evaluate 聚合。
- **外科变更专项测试**：
  - 断言 `import agenteval` 不触发 tracing/云/遥测导入（用 `sys.modules` 检查 `agenteval.tracing`/`posthog`/`opentelemetry` 未被加载）。
  - 断言无 model 参数时 `initialize_model()` 默认返回 `OpenAIModel` 且不 import 其他 provider 模块。
- **端到端**：`evaluate([LLMTestCase(input=..., actual_output=...)], [GEval(criteria=..., model=MockModel())])` 返回带分数与 success 的 `EvaluationResult`。
- 测试框架：pytest（作为 dev 依赖，不作为运行时硬依赖）。

## 8. Agent 扩展路径（决策点②：已定 —— v1 只留扩展位）

- **v1 交付**：单轮 `evaluate` 端到端可用（LLMTestCase + BaseMetric + GEval + OpenAIModel）。
- **扩展位就位**：`ConversationalTestCase`/`Turn`/`BaseConversationalMetric` 一并移植（依赖极少），但 v1 **不接 conversational 分支 metric**。等做 Agent 评估时：
  1. 补 `_a_execute_conversational_test_cases` 接线（同 asyncio 引擎，改动小）；
  2. 按需移植 agentic metric（ToolUse / GoalAccuracy / StepEfficiency 等）；
  3. 数据模型无需返工（`LLMTestCase` 已有 `tools_called`/`expected_tools`，`Turn` 已有 `tools_called`）。
- 单轮也能承载工具调用评估（`ToolCorrectnessMetric` 等 rule-based 可后续按需移植）。

## 9. 依赖清单

### 运行时硬依赖
- `pydantic>=2`
- `openai`（默认 provider，OpenAIModel 顶层 eager import）
- `jinja2`（GEval 提示词模板，决策点③：已定保留）
- `nest_asyncio`（async-in-sync 兼容）
- `tqdm`（进度，metric 指示器）
- `rich`（可选精简报告工具；evaluate 默认不打印，仅作可选输出工具）

### 可选运行时依赖（按需 lazy import）
- `pytest`（若保留 `assert_test` 便利方法 + pytest 集成）
- provider SDK：`anthropic`、`google-genai`、`litellm`、`ollama` 等（仅在用户构造对应 model 时 import）

### 彻底丢弃
`chromadb`、`langchain*`、`tiktoken`、`torch`/`transformers`/`nltk`/`detoxify`、`ragas`、`opentelemetry-*`、`posthog`、`grpcio`、`typer`/`click`（CLI）、`textual`、`pyfiglet`、`questionary`、`portalocker`、`tabulate`。

## 10. 公开 API（`agenteval/__init__.py`）

```python
from agenteval.errors import AgentEvalError
from agenteval.test_case import LLMTestCase, ToolCall, ConversationalTestCase, Turn
from agenteval.metrics import BaseMetric, GEval, ExactMatchMetric
from agenteval.models import DeepEvalBaseLLM, OpenAIModel
from agenteval.evaluate import evaluate, EvaluationResult, TestResult
```

## 11. 决策点小结

| 决策 | 选择 | 理由 |
|---|---|---|
| ① 类命名 | 保持 `DeepEval*` 原名 | 忠实 fork，减少跨引用改坏风险；仅包路径与基类异常改名 |
| ② conversational | v1 只留扩展位不接线 | 贴合"单轮起步"；数据模型已就位，v2 接线改动小 |
| ③ GEval 提示词 | 保留 jinja2 模板包 | GEval 是主力 metric，模板可覆盖是其核心能力；jinja2 轻量 |

## 12. 开放问题（实现阶段再定，不阻塞 spec）

- `evaluate()` 是否提供同步入口别名（如 `evaluate_sync`）还是仅靠 `run_async=False` 参数控制 —— 沿用 deepeval 的 `AsyncConfig` 即可，不新增 API。
- rich 报告：保留一个轻量 `print_results(result)` 工具函数（可选），默认不打印。
- 是否移植 `assert_test` 作为便利方法（单 case + 失败抛错）—— 列为 v1 可选收尾项，不阻塞主路径。
