"""Prompt templates for AIToneMetric (Jinja2 strings).

Each method renders a prompt via ``render_template``. Subclass this
and override any method to customize that prompt only.
"""

from agenteval.templates.render import render_template

_GENERATE_AI_TONE_SCORE = """你正在评估一段 AI 实际输出的"AI 味"（机器生成痕迹）有多重。请就 AI 味的有无给出严格二元判定，而非判断正确性或相关性。

浓重 AI 味的常见特征：
- 机械化的结构过渡：如"首先……其次……最后"、"第一……第二……"等刻板列举
- 空话套话：如"值得注意的是"、"综上所述"、"总而言之"、"在当今……背景下"等无意义的开场铺垫
- 浮夸、空泛的措辞；过度对冲；过度客套
- 模板化的对称、工整句式；列表式的排比
- 含糊笼统、空洞的总结，缺乏具体、扎实的内容
- 不自然的均衡与打磨，缺少人类语言的起伏与个性

只返回 1 或 0：
- 1 表示存在明显 AI 味（机器生成痕迹明显）
- 0 表示读起来像自然的人类表达（无明显 AI 味）
不要给出介于两者之间的分数。若该文本整体更像机器生成则给 1，更像自然人类表达则给 0。

{% if multimodal %}{{ _fragments.multimodal_input_rules }}{% endif %}

**
重要：请务必只返回合法、可解析的 JSON 格式，包含 'score' 键（只能取整数 1 或 0）和 'reason' 键。确保所有字符串正确闭合，输出前修复任何非法 JSON。

示例 JSON：
{
  "score": 1,
  "reason": "大量使用 首先/其次/最后 过渡，结尾以 综上所述 收束，模板化痕迹明显。"
}
===== 示例结束 =====
**

实际输出：
{{ actual_output }}

JSON："""


class AIToneTemplate:
    """Default prompt templates for ``AIToneMetric``."""

    @staticmethod
    def generate_ai_tone_score(**kwargs):
        return render_template(_GENERATE_AI_TONE_SCORE, **kwargs)
