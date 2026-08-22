"""Copy a deepeval source file into agenteval, rewriting deepeval->agenteval.

Usage: python scripts/rename_deepeval.py <src_abs_path> <dest_abs_path>
"""
import sys
from pathlib import Path

REPLACEMENTS = [
    ("from deepeval", "from agenteval"),
    ("import deepeval", "import agenteval"),
    ('"deepeval.', '"agenteval.'),
    ("deepeval.templates", "agenteval.templates"),
]


def main(src: str, dest: str) -> None:
    text = Path(src).read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # generic catch-all for any remaining bare `deepeval.` references
    text = text.replace("deepeval.", "agenteval.")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(text, encoding="utf-8")
    print(f"copied {src} -> {dest}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
