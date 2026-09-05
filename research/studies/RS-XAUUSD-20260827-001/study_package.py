#!/usr/bin/env python3
"""Write and register a study package, so a runner stops re-implementing it.

Twenty-five of the thirty-eight runners in this repository hand-roll the same four steps:
make the output directory, serialise a payload, write `results.json`, and — sometimes,
inconsistently — add a row to `research/registry.json`. That last one being optional is
how ten confirmed studies ended up unregistered and unpublished.

This module is deliberately small. It absorbs the mechanics that every study shares and
nothing that any study decides. What to measure, which screens to apply and what the
verdict means all stay in the runner, because those are the parts that differ and the
parts a reader has to be able to audit.

Existing runners are NOT migrated. Their numbers are published and a refactor underneath
them makes those numbers unreproducible for the sake of tidiness — the same reason
`build_xauusd_hypothesis_sweep.py` keeps its frozen copy of the screen battery. New
studies use this; the old ones stay as they are.

Usage from a runner:

    from study_package import write_package

    write_package(
        study_id="RS-XAUUSD-20260825-001",
        payload=results_dict,
        market="XAUUSD",
        strategy="none — regime classification over existing strategies",
        title="...",
        question="...",
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "research/registry.json"
TAIPEI = timezone(timedelta(hours=8))


def study_dir(study_id: str) -> Path:
    return ROOT / "research/studies" / study_id


def write_results(study_id: str, payload: dict[str, Any]) -> Path:
    """Write `results.json`, creating the package directory if it is new."""
    directory = study_dir(study_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "results.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def register(study_id: str, *, market: str, strategy: str, title: str,
             status: str = "progress", created_on: str | None = None) -> bool:
    """Add or refresh this study's row in the registry. Returns True if it changed.

    Registration is not optional here. A study that runs but never registers is invisible
    to `verify_research_registry.py`, to the null-results builder and to the Public
    exporter, which is exactly the state ten studies sat in.
    """
    data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    row = {
        "id": study_id,
        "market": market,
        "strategy": strategy,
        "title": title,
        "status": status,
        "created_on": created_on or datetime.now(TAIPEI).date().isoformat(),
        "private_path": f"research/studies/{study_id}",
        "public_path": None,
    }
    existing = next((s for s in data["studies"] if s["id"] == study_id), None)
    if existing is None:
        data["studies"].append(row)
    else:
        # Never clobber a public_path that the exporter set.
        row["public_path"] = existing.get("public_path")
        row["created_on"] = existing.get("created_on", row["created_on"])
        if existing == row:
            return False
        data["studies"][data["studies"].index(existing)] = row
    data["studies"].sort(key=lambda s: s["id"])
    REGISTRY.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    return True


def write_study_json(study_id: str, *, market: str, strategy: str, title: str,
                     question: str, hypothesis: str, runner: str,
                     headline: dict[str, Any], findings: list[dict[str, str]],
                     card_summary: str, theme: str = "methodology",
                     status: str = "progress",
                     limitations: list[str] | None = None,
                     card_metrics: list[str] | None = None) -> Path:
    """Write `study.json` with the fields the verifier and the Public exporter require."""
    directory = study_dir(study_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "study.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    document = {
        "schema_version": 1,
        "id": study_id,
        "revision": existing.get("revision", 1),
        "title": title,
        "market": market,
        "strategy": strategy,
        "status": status,
        "created_on": existing.get("created_on",
                                   datetime.now(TAIPEI).date().isoformat()),
        "question": question,
        "hypothesis": hypothesis,
        "runner": runner,
        "results": f"research/studies/{study_id}/results.json",
        "source_manifest": f"research/studies/{study_id}/source_manifest.json",
        "decision_log": f"research/studies/{study_id}/decision_log.md",
        "card_metrics": card_metrics or list(headline)[:4],
        "headline": headline,
        "findings": findings,
        "policy_impacts": existing.get("policy_impacts", []),
        "impact_doc": existing.get("impact_doc"),
        # A new study is never born reviewed. The Public gate stays shut until a person
        # has read it.
        "public_release": existing.get("public_release", {
            "status": "pending",
            "raw_csv_included": False,
            "private_conversation_included": False,
            "aggregate_results_included": True,
            "method_included": True,
            "charts_included": False,
        }),
        "limitations": limitations,
        "theme": theme,
        "card_summary": card_summary,
    }
    # 2026-09-05 independent review: this function rebuilt the whole document and carried
    # only five keys over from `existing`, so re-running any runner silently deleted every
    # hand-authored field the runner does not produce -- the `presentation` blueprint, the
    # Chinese narrative (`interpretation_zh`, `limitations_zh`, `question_zh`, the per-
    # finding `title_zh`/`detail_zh`), and, for a runner that passes `findings=[]`, the
    # curated findings themselves. That is exactly what happened to
    # RS-XAUUSD-20260825-001 on 2026-09-05: its reader page lost its presentation blocks
    # and the Public build crashed on the missing key. Restoring the artifact by hand fixed
    # that one study and left the cause in place, so this now preserves anything curated.
    #
    # The rule: a key the runner did not compute is the author's, not the runner's. Never
    # let a rerun be a deletion.
    for key, value in existing.items():
        if key not in document:
            document[key] = value
    # The same rule applied to the three fields a runner *does* supply but an author
    # routinely enriches afterwards. A runner computes numbers; a person decides what the
    # study says and which numbers the card shows.
    #   headline     - merged: the runner's computed keys win, hand-added keys survive.
    #                  RS-XAUUSD-20260825-001 carried best_candidate / worst_family_p /
    #                  best_candidate_trades_needed here; a rerun deleted all four.
    #   card_metrics - the curated choice of which headline keys to show wins over the
    #                  runner's `list(headline)[:4]` default, which is a fallback, not a
    #                  decision.
    #   card_summary - published prose is a research conclusion; CLAUDE.md puts changing
    #                  one behind explicit owner scope, so a rerun never rewrites it.
    # To genuinely replace any of these, edit study.json (or drop the field) -- an explicit
    # act by a person, which is the point.
    if isinstance(existing.get("headline"), dict):
        for key, value in existing["headline"].items():
            document["headline"].setdefault(key, value)
    if existing.get("card_metrics"):
        document["card_metrics"] = existing["card_metrics"]
    if existing.get("card_summary"):
        document["card_summary"] = existing["card_summary"]
    # `findings` is a parameter, so an empty list from a runner that does not author them
    # would still overwrite curated ones. An empty result never replaces a non-empty record.
    if not findings and existing.get("findings"):
        document["findings"] = existing["findings"]
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def write_package(study_id: str, payload: dict[str, Any], **study_fields: Any) -> dict:
    """Write results.json, study.json and the registry row in one call."""
    results = write_results(study_id, payload)
    manifest = study_fields.pop("study_json", True)
    written = {"results": str(results.relative_to(ROOT))}
    if manifest:
        study = write_study_json(study_id, **study_fields)
        written["study"] = str(study.relative_to(ROOT))
    written["registry_changed"] = register(
        study_id,
        market=study_fields["market"],
        strategy=study_fields["strategy"],
        title=study_fields["title"],
        status=study_fields.get("status", "progress"),
    )
    return written
