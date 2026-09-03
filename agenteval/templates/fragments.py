"""Shared prompt fragments for Jinja2 metric templates.

Inject these via ``render_template`` (the ``_fragments`` name is reserved).
Only ``multimodal_input_rules`` is referenced by implemented metrics; add
more here if a future metric needs a shared snippet.
"""

FRAGMENTS = {
    "multimodal_input_rules": """
--- 多模态输入规则 ---
- 将图像内容视为事实证据。
- 仅引用清晰可见的视觉细节。
- 不要推断或猜测图像中未明确呈现的对象、文字或细节。
- 若图像模糊或存在歧义，请明确标注不确定性。
""",
}
