from __future__ import annotations

import csv
import difflib
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests


ROOT = Path("/data/lm/building-data-bdg2")
BIB_PATH = ROOT / "paper/latex/paper/references.bib"
NOCITE_PATH = ROOT / "paper/latex/paper/sections/09_references.tex"
OUT_PATH = ROOT / "tables/reference_verification_crossref.csv"


def parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    starts = [m.start() for m in re.finditer(r"(?m)^@", text)]
    starts.append(len(text))
    for start, end in zip(starts, starts[1:]):
        block = text[start:end].strip()
        if not block:
            continue
        brace = block.find("{")
        if brace == -1:
            continue
        first_comma = block.find(",", brace)
        if first_comma == -1:
            continue
        entry_type = block[1:brace].strip()
        key = block[brace + 1 : first_comma].strip()
        body = block[first_comma + 1 : -1]
        fields: dict[str, str] = {"ENTRYTYPE": entry_type}
        for match in re.finditer(r"(?m)^\s*([A-Za-z]+)\s*=\s*\{(.*?)\},?\s*$", body, re.S):
            fields[match.group(1).lower()] = match.group(2).strip()
        entries[key] = fields
    return entries


def selected_keys(text: str) -> list[str]:
    m = re.search(r"\\nocite\{([^}]*)\}", text, re.S)
    if not m:
        return []
    return [k.strip() for k in m.group(1).split(",") if k.strip()]


def normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("---", " ").replace("--", " ").replace("-", " ")
    text = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\&", "&").replace(r"\i{}", "i")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def year_from_item(item: dict) -> str:
    for key in ("published-print", "published-online", "issued", "created"):
        value = item.get(key)
        if value and value.get("date-parts"):
            return str(value["date-parts"][0][0])
    return ""


def verify_with_crossref(session: requests.Session, title: str) -> dict[str, str]:
    url = "https://api.crossref.org/works?rows=5&query.title=" + urllib.parse.quote(title)
    response = session.get(
        url,
        headers={"User-Agent": "Codex reference verifier/1.0 (mailto:author@example.com)"},
        timeout=6,
    )
    response.raise_for_status()
    items = response.json()["message"]["items"]
    best_item = None
    best_score = -1.0
    for item in items:
        item_title = (item.get("title") or [""])[0]
        score = difflib.SequenceMatcher(None, normalize(title), normalize(item_title)).ratio()
        if score > best_score:
            best_score = score
            best_item = item
    if best_item is None:
        return {
            "status": "no_match",
            "score": "",
            "matched_title": "",
            "matched_year": "",
            "matched_venue": "",
            "doi": "",
        }
    matched_title = (best_item.get("title") or [""])[0]
    matched_venue = (best_item.get("container-title") or [""])[0]
    return {
        "status": "exact"
        if normalize(title) == normalize(matched_title)
        else "high"
        if best_score >= 0.9
        else "medium"
        if best_score >= 0.75
        else "low",
        "score": f"{best_score:.3f}",
        "matched_title": matched_title,
        "matched_year": year_from_item(best_item),
        "matched_venue": matched_venue,
        "doi": best_item.get("DOI", ""),
    }


def main() -> int:
    bib = parse_bibtex(BIB_PATH.read_text())
    keys = selected_keys(NOCITE_PATH.read_text())

    existing: dict[str, dict[str, str]] = {}
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for row in csv.DictReader(f):
                existing[row["key"]] = row

    session = requests.Session()
    rows: list[dict[str, str]] = list(existing.values())
    done = set(existing)

    for index, key in enumerate(keys, 1):
        if key in done:
            continue
        fields = bib[key]
        title = fields.get("title", "")
        year = fields.get("year", "")
        venue = fields.get("journal", "") or fields.get("booktitle", "") or fields.get("note", "")
        print(f"[{index}/{len(keys)}] {key}: {title}", flush=True)
        try:
            result = verify_with_crossref(session, title)
        except Exception as exc:
            result = {
                "status": "error",
                "score": "",
                "matched_title": "",
                "matched_year": "",
                "matched_venue": "",
                "doi": "",
                "error": repr(exc),
            }
        row = {
            "key": key,
            "bib_title": title,
            "bib_year": year,
            "bib_venue": venue,
            "status": result.get("status", ""),
            "title_score": result.get("score", ""),
            "matched_title": result.get("matched_title", ""),
            "matched_year": result.get("matched_year", ""),
            "matched_venue": result.get("matched_venue", ""),
            "doi": result.get("doi", ""),
            "error": result.get("error", ""),
        }
        rows.append(row)
        with OUT_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "key",
                    "bib_title",
                    "bib_year",
                    "bib_venue",
                    "status",
                    "title_score",
                    "matched_title",
                    "matched_year",
                    "matched_venue",
                    "doi",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        time.sleep(0.2)

    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
