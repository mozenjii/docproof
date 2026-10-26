"""Source policy: document data reaches output only through docproof.safelog.

This runs over the package source, so the rule holds for code written later, not just today's.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "docproof"
SOURCES = sorted(PACKAGE.rglob("*.py"))
LOGGING_OWNER = PACKAGE / "safelog.py"
BANNED_MODULES = {"logging", "traceback", "pprint"}


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _prints_outside_main(tree: ast.AST) -> list[int]:
    """Line numbers of print() calls not inside a function called main."""
    offending: list[int] = []

    def visit(node: ast.AST, in_main: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inside = in_main
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                inside = child.name == "main"
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "print"
                and not inside
            ):
                offending.append(child.lineno)
            visit(child, inside)

    visit(tree, in_main=False)
    return offending


def test_the_policy_sees_the_package() -> None:
    assert LOGGING_OWNER in SOURCES
    assert len(SOURCES) > 10


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_only_safelog_imports_output_modules(path: Path) -> None:
    banned = _imports(ast.parse(path.read_text(encoding="utf-8"))) & BANNED_MODULES
    if path == LOGGING_OWNER:
        banned -= {"logging"}
    assert not banned, f"{path.name} imports {sorted(banned)}; log through docproof.safelog"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.relative_to(PACKAGE).as_posix())
def test_print_only_in_command_line_entry_points(path: Path) -> None:
    lines = _prints_outside_main(ast.parse(path.read_text(encoding="utf-8")))
    assert not lines, f"{path.name} prints outside main() at lines {lines}"


def test_the_checks_catch_a_violation() -> None:
    bad = ast.parse("import logging\nfrom traceback import format_exc\ndef f():\n    print(1)\n")
    assert _imports(bad) >= {"logging", "traceback"}
    assert _prints_outside_main(bad) == [4]
    good = ast.parse("def main():\n    print('wrote 3 documents')\n")
    assert _prints_outside_main(good) == []
