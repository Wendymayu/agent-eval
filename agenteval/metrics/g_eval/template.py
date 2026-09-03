"""Prompt templates for GEval (Jinja2 strings).

Each method renders a prompt via ``render_template``. Subclass this
and override any method to customize that prompt only.
"""

from agenteval.templates.render import render_template

_GENERATE_EVALUATION_RESULTS = """你是一名评估者。根据下面的{% if rubric %}评估步骤与评分细则{% else %}评估步骤{% endif %}，对下方的回答进行评估，并返回一个包含两个字段的 JSON 对象：

- `"score"`：一个介于 {{ score_range[0] }} 与 {{ score_range[1] }} 之间的整数，{% if rubric %}依据所给评分细则{% else %}其中 {{ score_range[1] }} 表示与评估步骤高度吻合，{{ score_range[0] }} 表示完全不吻合{% endif %}。
- `"reason"`：简要说明给出该分数的理由。须提及具体的优点或不足，并引用输入中的相关细节。解释中**不要**引用分数本身。

你的解释应当：
- {% if rubric %}具体且立足于评估步骤与评分细则。{% else %}具体且立足于评估步骤。{% endif %}
- 提及测试用例参数中的关键细节。
- 简洁、清晰、聚焦于评估逻辑。
{% if multimodal %}{{ _fragments.multimodal_input_rules }}{% endif %}

只返回合法 JSON。**不要**包含任何额外的评论或文字。

---

评估步骤：
{{ evaluation_steps }}

{% if rubric %}评分细则：
{{ rubric }}

{% endif %}测试用例：
{{ test_case_content }}

参数：
{{ parameters }}
{% if _additional_context %}

补充上下文：
{{ _additional_context }}
{% endif %}

---
**示例 JSON：**
{
  "reason": "你的简洁且信息丰富的理由",
  "score": {{ score_range[0] }}
}

JSON：
"""

_GENERATE_EVALUATION_STEPS = """给定一组评估标准，说明应如何评判 {{ parameters }}，请基于下面的标准生成 3-4 条简洁的评估步骤。你必须清楚说明如何相对于彼此来评判 {{ parameters }}。

{% if multimodal %}{{ _fragments.multimodal_input_rules }}{% endif %}

评估标准：
{{ criteria }}

**
重要：请务必只返回 JSON 格式，其中 "steps" 键为字符串列表。无需任何多余的文字或解释。
示例 JSON：
{
  "steps": <字符串列表>
}
**

JSON：
"""

_GENERATE_STRICT_EVALUATION_RESULTS = """根据评估步骤，返回一个包含两个键的 JSON：1) `score` 键，取值严格只能是 1（完全符合评估步骤所述标准）或 0（不符合标准）；2) `reason` 键，给出该分数的理由，但**不要**在理由中引用分数。请在理由中提及 {{ parameters }} 中的具体信息，但要非常简洁！

{% if multimodal %}{{ _fragments.multimodal_input_rules }}{% endif %}

评估步骤：
{{ evaluation_steps }}

{{ test_case_content }}
{% if _additional_context %}

补充上下文：
{{ _additional_context }}
{% endif %}
**
重要：请务必只返回 JSON 格式，包含 "score" 和 "reason" 键。无需任何多余的文字或解释。

示例 JSON：
{
  "reason": "该文本不符合所给的评估步骤。",
  "score": 0
}
**

JSON：
"""

class GEvalTemplate:
    """Default prompt templates for ``GEval``."""

    @staticmethod
    def generate_evaluation_results(**kwargs):
        return render_template(_GENERATE_EVALUATION_RESULTS, **kwargs)

    @staticmethod
    def generate_evaluation_steps(**kwargs):
        return render_template(_GENERATE_EVALUATION_STEPS, **kwargs)

    @staticmethod
    def generate_strict_evaluation_results(**kwargs):
        return render_template(_GENERATE_STRICT_EVALUATION_RESULTS, **kwargs)
