"""Static workload-family inference from task text. The text is parsed, never executed."""

import ast
import re
import sys

FAMILIES: tuple[str, ...] = ("FUNC", "LIB", "ALGO_FN", "ALGO_IO", "CLASS")
STDLIB_MODULES: frozenset[str] = frozenset(sys.stdlib_module_names)


def static_ast_stats(prompt_text: str) -> dict[str, int | bool]:
    """Static-only AST inspection of code blocks in a prompt (never executes)."""
    code_candidate = prompt_text
    if "```python" in prompt_text:
        after = prompt_text.split("```python", 1)[1]
        code_candidate = after.split("```", 1)[0]
    try:
        tree = ast.parse(code_candidate)
    except SyntaxError:
        return {
            "ast_nodes": 0,
            "n_functions": len(re.findall(r"\bdef\s+\w+", prompt_text)),
            "n_classes": len(re.findall(r"\bclass\s+\w+", prompt_text)),
            "n_branches": len(re.findall(r"\b(?:if|elif)\b", prompt_text)),
            "n_loops": len(re.findall(r"\b(?:for|while)\b", prompt_text)),
            "third_party": False,
        }
    third_party = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in STDLIB_MODULES:
                    third_party = True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] not in STDLIB_MODULES:
                third_party = True
    return {
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "n_functions": sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree)
        ),
        "n_classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        "n_branches": sum(isinstance(n, ast.If) for n in ast.walk(tree)),
        "n_loops": sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in ast.walk(tree)),
        "third_party": third_party,
    }


def infer_prompt_family(prompt_text: str) -> str:
    """Infer coding workload family purely from static prompt text & AST (never executes)."""
    lowered = prompt_text.lower()
    if "stdin" in lowered or "standard input" in lowered or "sample input" in lowered:
        return "ALGO_IO"
    stats = static_ast_stats(prompt_text)
    if int(stats["n_classes"]) > 0 or "implement every method" in lowered:
        return "CLASS"
    if bool(stats["third_party"]):
        return "LIB"
    return "FUNC"
