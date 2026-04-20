#!/usr/bin/env python3
"""Patch ADAPTed's SigProcConfig to use `field(default_factory=...)`.

ADAPTed (KleistLab/ADAPTed) declares `@dataclass` fields with mutable
class-instance defaults:

    @dataclass
    class SigProcConfig(NestedConfig):
        core: CoreConfig = CoreConfig()
        ...

Python dataclasses have rejected this pattern for years, but the upstream
repo hasn't released a fix. We rewrite the problem block in place to:

    core: CoreConfig = field(default_factory=CoreConfig)

Idempotent: leaves already-patched files untouched.

Used by the `install-warpdemux` pixi task before `pip install -e ext/ADAPTed`.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FIELD_RE = re.compile(
    r"^(?P<indent>\s+)(?P<name>\w+)\s*:\s*(?P<ty>\w+)\s*=\s*(?P=ty)\(\s*\)\s*$",
    re.MULTILINE,
)


def patch(path: Path) -> bool:
    src = path.read_text()
    if "field(default_factory=" in src and "CoreConfig()" not in src:
        return False  # already patched

    def sub(match: re.Match) -> str:
        return f"{match['indent']}{match['name']}: {match['ty']} = field(default_factory={match['ty']})"

    new = FIELD_RE.sub(sub, src)
    if new == src:
        return False

    # Ensure `field` is imported from `dataclasses`.
    if "from dataclasses import" in new and "field" not in new.split("from dataclasses import", 1)[1].split("\n", 1)[0]:
        new = re.sub(
            r"from dataclasses import ([^\n]+)",
            lambda m: f"from dataclasses import {m.group(1).rstrip()}, field",
            new,
            count=1,
        )
    elif "from dataclasses import" not in new:
        new = "from dataclasses import field\n" + new

    path.write_text(new)
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: patch_adapted_dataclass.py <file-or-dir>", file=sys.stderr)
        return 2
    target = Path(sys.argv[1])
    if not target.exists():
        print(f"not found: {target}", file=sys.stderr)
        return 1
    files = [target] if target.is_file() else sorted(target.rglob("*.py"))
    for f in files:
        changed = patch(f)
        print(f"{'patched' if changed else 'unchanged'}: {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
