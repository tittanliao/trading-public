#!/usr/bin/env python3
"""Build deterministic landing pages from reviewed Public repository contents."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "04cd05734e6905561e113945948e848e106d26bb"
GENERATED = {
    Path("index.html"),
    Path("xauusd/index.html"),
    Path("tx/index.html"),
    Path("research/index.html"),
    Path("site/catalog.json"),
}
EXCLUDED_PARTS = {".git", ".github", "legacy-site", "site", "__pycache__"}


def title_for(path: Path) -> str:
    if path.suffix.lower() == ".html":
        text = path.read_text(encoding="utf-8", errors="ignore")[:100_000]
        match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        if match:
            return re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip()


def section_for(relative: Path) -> str:
    if relative.parts and relative.parts[0] == "xauusd":
        return "xauusd"
    if relative.parts and relative.parts[0] == "tx":
        return "tx"
    return "research"


def catalog() -> dict[str, object]:
    items: list[dict[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative in GENERATED or EXCLUDED_PARTS.intersection(relative.parts):
            continue
        extension = path.suffix.lower()
        if extension not in {".html", ".pine", ".py", ".json"}:
            continue
        if extension == ".html":
            kind = "report"
        elif extension == ".pine":
            kind = "pine"
        elif extension == ".py":
            kind = "python"
        else:
            kind = "result"
        items.append(
            {
                "path": relative.as_posix(),
                "title": title_for(path),
                "section": section_for(relative),
                "kind": kind,
            }
        )
    return {"schema_version": 1, "legacy_source_commit": SOURCE_COMMIT, "items": items}


def nav(prefix: str) -> str:
    return (
        '<nav class="nav">'
        f'<a href="{prefix}index.html">Overview</a>'
        f'<a href="{prefix}xauusd/">XAUUSD</a>'
        f'<a href="{prefix}tx/">TX</a>'
        f'<a href="{prefix}research/">Research files</a>'
        '</nav>'
    )


def document(title: str, eyebrow: str, lede: str, body: str, prefix: str = "") -> str:
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="{html.escape(lede)}">
  <title>{html.escape(title)} · Trading Research</title>
  <link rel="stylesheet" href="{prefix}site/style.css">
</head>
<body>
  <header class="shell">
    <div class="eyebrow">{html.escape(eyebrow)}</div>
    <h1>{html.escape(title)}</h1>
    <p class="lede">{html.escape(lede)}</p>
    {nav(prefix)}
  </header>
  {body}
  <footer><div class="shell">Generated from reviewed repository contents. Research evidence, not trading advice.</div></footer>
  <script src="{prefix}site/app.js"></script>
</body>
</html>
"""


def card(item: dict[str, str], prefix: str) -> str:
    path = html.escape(prefix + item["path"])
    return (
        f'<a class="card" data-card href="{path}">'
        f'<div class="type">{html.escape(item["kind"])}</div>'
        f'<h3>{html.escape(item["title"])}</h3>'
        f'<p>{html.escape(item["path"])}</p></a>'
    )


def section_page(data: dict[str, object], section: str, title: str, lede: str) -> str:
    items = [item for item in data["items"] if item["section"] == section]
    body = (
        '<div class="toolbar"><div class="shell"><input data-search type="search" '
        'placeholder="Filter by title or path" aria-label="Filter"></div></div>'
        f'<main class="shell"><div class="stats"><span class="stat"><strong>{len(items)}</strong> indexed files</span></div>'
        f'<div class="grid">{"".join(card(item, "../") for item in items)}</div></main>'
    )
    return document(title, section, lede, body, "../")


def overview(data: dict[str, object]) -> str:
    items = data["items"]
    counts = {
        section: sum(item["section"] == section for item in items)
        for section in ("xauusd", "tx", "research")
    }
    cards = [
        ("xauusd/", "XAUUSD", "Gold strategy reports, Pine sources, and supporting research.", counts["xauusd"]),
        ("tx/", "TX", "Taiwan index futures experiments, reports, and code.", counts["tx"]),
        ("research/", "Research files", "Cross-market utilities, result JSON, and reproducible scripts.", counts["research"]),
        ("legacy-site/index.html", "Legacy site snapshot", "Archived mixed landing page for comparison; not the active workflow.", 6),
    ]
    body = '<main class="shell"><div class="grid">' + "".join(
        f'<a class="card" href="{href}"><div class="type">collection</div><h2>{title}</h2><p>{description} · {count} items</p></a>'
        for href, title, description, count in cards
    ) + '</div></main>'
    return document(
        "Trading research, without the maze.",
        "Reviewed public workspace",
        "A generated portal for published reports, code, and strategy evidence. Raw data and private model memory stay outside this repository.",
        body,
    )


def outputs(data: dict[str, object]) -> dict[Path, str]:
    return {
        ROOT / "site/catalog.json": json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        ROOT / "index.html": overview(data),
        ROOT / "xauusd/index.html": section_page(data, "xauusd", "XAUUSD research", "Published gold strategy reports, Pine sources, and supporting evidence."),
        ROOT / "tx/index.html": section_page(data, "tx", "TX research", "Published Taiwan index futures experiments, reports, and code."),
        ROOT / "research/index.html": section_page(data, "research", "Research files", "Cross-market utilities, result files, and reproducible source code."),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = outputs(catalog())
    failures: list[str] = []
    for path, expected in generated.items():
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                failures.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8")
    print(f"generated files: {len(generated)}")
    print(f"drift: {len(failures)}")
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
