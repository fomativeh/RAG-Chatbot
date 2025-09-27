from typing import List
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "prompts"

def _read_template(filename: str) -> str:
    path = _BASE / filename
    return path.read_text(encoding="utf-8")


def build_system_prompt(mode: str = "default", repo_url: str | None = None, docs_dir: str = "data/default_docs") -> str:
    fallback = (
        "The data I am trained on doesn't provide an answer to that question."
        if mode == "default"
        else "I don't know."
    )
    template = _read_template("system_default.txt" if mode == "default" else "system_custom.txt")
    return template.format(fallback=fallback, repo_url=(repo_url or ""), docs_dir=docs_dir)


def format_context(chunks: List[str], labels: List[str]) -> str:
    formatted = []
    for i, (chunk, label) in enumerate(zip(chunks, labels)):
        formatted.append(f"[{label}]\n{chunk}")
    return "\n\n".join(formatted)


def build_user_prompt(question: str, context_block: str, mode: str = "default", repo_url: str | None = None, docs_dir: str = "data/default_docs") -> str:
    fallback = (
        "The data I am trained on doesn't provide an answer to that question."
        if mode == "default"
        else "I don't know."
    )
    template = _read_template("user_default.txt" if mode == "default" else "user_custom.txt")
    has_context = bool((context_block or "").strip())
    return template.format(
        question=question,
        context_block=context_block,
        fallback=(fallback if not has_context else ""),
        repo_url=(repo_url or ""),
        docs_dir=docs_dir,
    )


def build_safety_check_prompt(question: str, draft_answer: str, context_block: str) -> str:
    template = _read_template("safety.txt")
    return template.format(question=question, context_block=context_block, draft_answer=draft_answer)


