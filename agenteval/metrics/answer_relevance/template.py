"""Prompt templates for AnswerRelevanceMetric (Jinja2 strings).

Each method renders a prompt via ``render_template``. Subclass this
and override any method to customize that prompt only.
"""

from agenteval.templates.render import render_template

_GENERATE_RELEVANCE_SCORE = """你正在评估 AI 的实际输出与给定输入的相关性。若输出回应或针对了输入，则视为相关；若输出忽略、回避或谈论与输入无关的内容，则视为不相关。

请就输入与实际输出，给出严格二元的相关性判定。只返回 1 或 0：
- 1 表示输出切中、扣题地回应了输入（相关）
- 0 表示输出忽略、回避或离题（不相关）
不要给出介于两者之间的分数。若输出已实质回应输入则给 1，若回避或离题则给 0。

{% if multimodal %}{{ _fragments.multimodal_input_rules }}{% endif %}

**
重要：请务必只返回合法、可解析的 JSON 格式，包含 'score' 键（只能取整数 1 或 0）和 'reason' 键。确保所有字符串正确闭合，输出前修复任何非法 JSON。

示例 JSON：
{
  "score": 1,
  "reason": "输出直接回应了输入所问。"
}
===== 示例结束 =====
**

输入：
{{ input }}

实际输出：
{{ actual_output }}

JSON："""


class AnswerRelevanceTemplate:
    """Default prompt templates for ``AnswerRelevanceMetric``."""

    @staticmethod
    def generate_relevance_score(**kwargs):
        return render_template(_GENERATE_RELEVANCE_SCORE, **kwargs)
