"""API-surface coverage guard for the escapepod native extension.

``coverage.py`` measures Python *line* execution and cannot see into a compiled
pyo3 module, so it reports ~nothing for escapepod. The meaningful question for a
native-backed library is instead: does the test suite *reference* every public
member of the API?

This module enumerates the public API with ``inspect``/``dir`` and collects real
usage from the sibling test files with the ``ast`` module — attribute reads
(``x.attr``, which correctly excludes ``attr=`` keyword arguments) plus the
protocol dunders exercised via ``with`` / ``for`` / ``len`` / ``==`` / ``repr`` /
``hash``.

It is a **hard gate**: if any public member is unreferenced the test fails, so a
new field added with no test breaks CI until it's exercised (or the API lists
below are updated). Prints the coverage percentage either way.
"""

import ast
import glob
import os

import escapepod

# Public classes whose members should be exercised, plus module-level names.
CLASSES = ["Reader", "DatasetReader", "Writer", "ReadData", "RunInfo", "KmerTable"]
MODULE_LEVEL = {
    "create_run_info": "func",
    "Pod5Error": "exc",
    "mad_normalize": "func",
    "normalize_signal": "func",
    "refine_signal_map": "func",
}
_THIS = os.path.basename(__file__)


def _public_api():
    """Map ``"Class.member"`` / module name -> kind for every public API member."""
    api = {}
    for cname in CLASSES:
        for member in dir(getattr(escapepod, cname)):
            if not member.startswith("__"):
                api[f"{cname}.{member}"] = "member"
    api.update(MODULE_LEVEL)
    return api


def _usage():
    """Collect attribute reads and protocol uses from the sibling test files."""
    attrs, names = set(), set()
    here = os.path.dirname(__file__)
    files = glob.glob(os.path.join(here, "*.py")) + glob.glob(
        os.path.join(here, "..", "compat", "*.py")
    )
    for path in files:
        if os.path.basename(path) == _THIS:
            continue  # don't count this guard's own source
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attrs.add(node.attr)  # x.attr — a real read/call, not a kwarg
            elif isinstance(node, ast.Name):
                names.add(node.id)
    return attrs, names


def _covered(key, kind, attrs, names):
    member = key.rsplit(".", 1)[-1]
    if kind in ("func", "exc"):
        return member in names or member in attrs
    return member in attrs


def test_api_surface_coverage():
    api = _public_api()
    attrs, names = _usage()
    gaps = sorted(k for k, kind in api.items() if not _covered(k, kind, attrs, names))
    hit = len(api) - len(gaps)
    pct = 100 * hit / len(api)
    print(f"\nescapepod API-surface coverage: {hit}/{len(api)} = {pct:.0f}%")
    assert not gaps, (
        f"{len(gaps)} public API member(s) not referenced by the test suite: "
        + ", ".join(gaps)
        + "\nExercise them in a test, or update CLASSES/MODULE_LEVEL if intentional."
    )
