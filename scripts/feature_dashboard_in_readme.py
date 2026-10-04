#!/usr/bin/env python3
"""Insert / refresh a prominent "Live dashboard" banner in README.md.

The hosted dashboard URL depends on the GitHub owner/repo, which isn't known
when the workspace is scaffolded — so the publish workflow calls this on every
publish to keep a banner (between the dashboard markers) pointing at the live
site. If the markers are absent, the banner is inserted right after the first
H1 heading. No-op when the README is already current.

Usage:
    feature_dashboard_in_readme.py <readme> <dashboard_url> <reports_url>
"""
from __future__ import annotations

import sys
from pathlib import Path

BEGIN = "<!-- BEGIN dashboard -->"
END = "<!-- END dashboard -->"


def _block(dashboard_url: str, reports_url: str) -> str:
    return (
        f"{BEGIN}\n"
        f"> ## 📊 [**Live dashboard →**]({dashboard_url})\n"
        f"> Browse every investigation & study interactively, or read the "
        f"[published investigation reports]({reports_url}). "
        f"Auto-published from `main` on every merge.\n"
        f"{END}"
    )


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: feature_dashboard_in_readme.py <readme> <dashboard_url> <reports_url>",
              file=sys.stderr)
        return 2
    readme = Path(sys.argv[1])
    block = _block(sys.argv[2], sys.argv[3])
    if not readme.is_file():
        print(f"{readme} not found; nothing to do")
        return 0
    text = readme.read_text(encoding="utf-8")

    if BEGIN in text and END in text:
        i = text.index(BEGIN)
        j = text.index(END) + len(END)
        new = text[:i] + block + text[j:]
    else:
        # Insert right after the first H1 (`# Title`) line.
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for ln in lines:
            out.append(ln)
            if not inserted and ln.lstrip().startswith("# "):
                out.append("\n" + block + "\n")
                inserted = True
        new = "".join(out) if inserted else block + "\n\n" + text

    if new != text:
        readme.write_text(new, encoding="utf-8")
        print("README dashboard banner updated")
    else:
        print("README dashboard banner already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
