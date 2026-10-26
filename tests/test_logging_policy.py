"""Source policy: document data reaches output only through docproof.safelog.

This scans the package source, so the rule holds for code written later, not just today's. It is a
static check over names, so a determined contributor can still get around it (``getattr`` with a
built string, for one); what it guarantees is that every ordinary way of writing to a log or a
stream outside safelog fails the build and has to be argued for. Exceptions raised by docproof can
quote values in their messages; command-line entry points must catch them and log through
``SafeLogger.error``, which drops the message.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "docproof"
SOURCES = sorted(PACKAGE.rglob("*.py"))
LOGGING_OWNER = PACKAGE / "safelog.py"
BANNED_MODULES = {
    "logging",
    "traceback",
    "pprint",
    "warnings",
    "importlib",
    "builtins",
    "faulthandler",
}
STREAMS = {"stdout", "stderr", "__stdout__", "__stderr__"}


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _top_level_main(tree: ast.Module) -> set[int]:
    """Ids of nodes inside the module-level ``main`` function (not a method called main)."""
    inside: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "main":
            inside.update(id(n) for n in ast.walk(node))
    return inside


def _violations(tree: ast.Module) -> list[str]:
    """Every output route used outside the module-level main(), as 'line: what'."""
    in_main = _top_level_main(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "print" and id(node) not in in_main:
            found.append(f"{node.lineno}: print outside main()")
        elif isinstance(node, ast.Name) and node.id == "__import__":
            found.append(f"{node.lineno}: __import__")
        elif isinstance(node, ast.Attribute):
            on_sys = isinstance(node.value, ast.Name) and node.value.id == "sys"
            if on_sys and node.attr in STREAMS and id(node) not in in_main:
                found.append(f"{node.lineno}: sys.{node.attr} outside main()")
            if node.attr in {"logging", "_logging"}:
                found.append(f"{node.lineno}: reaches a logging module through an attribute")
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            streams = [a.name for a in node.names if a.name in STREAMS]
            if streams:
                found.append(f"{node.lineno}: imports {streams} from sys")
        elif isinstance(node, ast.ImportFrom) and node.module == "docproof.safelog":
            private = [a.name for a in node.names if a.name.startswith("_") or a.name == "logging"]
            if private:
                found.append(f"{node.lineno}: imports {private} from docproof.safelog")
    return found


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
def test_no_other_output_routes(path: Path) -> None:
    found = _violations(ast.parse(path.read_text(encoding="utf-8")))
    if path == LOGGING_OWNER:
        found = [f for f in found if "attribute" not in f]  # safelog uses its own _logging
    assert not found, f"{path.name}: {found}"


BYPASSES = {
    "from docproof.safelog import _logging": "imports",
    "import docproof.safelog as s\ns._logging.getLogger('x')": "attribute",
    "__import__('logging')": "__import__",
    "import sys\nsys.stderr.write('x')": "sys.stderr",
    "from sys import stdout": "from sys",
    "p = print": "print outside",
    "import builtins": "builtins",
    "import importlib": "importlib",
    "import warnings": "warnings",
    "class C:\n    def main(self):\n        print('x')": "print outside",
}


@pytest.mark.parametrize("source", list(BYPASSES), ids=list(BYPASSES.values()))
def test_the_policy_catches_known_bypasses(source: str) -> None:
    tree = ast.parse(source)
    reasons = " ".join(_violations(tree)) + " " + " ".join(sorted(_imports(tree)))
    assert BYPASSES[source] in reasons


def test_a_subprocess_result_is_not_a_stream() -> None:
    tree = ast.parse(
        "import subprocess\nout = subprocess.run(['git'], capture_output=True).stdout\n"
    )
    assert _violations(tree) == []


def test_the_policy_allows_a_cli_main() -> None:
    tree = ast.parse("import sys\ndef main():\n    print('ok')\n    sys.stderr.write('x')\n")
    assert _violations(tree) == []
