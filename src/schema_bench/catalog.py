from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from schema_bench.paths import CATALOG_PATH


@dataclass(frozen=True)
class SchemaDoc:
    id: str
    kind: str
    name: str
    description: str
    tags: tuple[str, ...]
    parent_table: str | None
    text: str


def _build_text(entry: dict) -> str:
    tags = " ".join(entry.get("tags") or [])
    parent = entry.get("parent_table") or ""
    parts = [
        entry["name"],
        entry["kind"],
        entry.get("description") or "",
        tags,
        parent,
    ]
    return " ".join(p for p in parts if p).strip()


def load_catalog(path: Path | None = None) -> list[SchemaDoc]:
    catalog_path = path or CATALOG_PATH
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    docs: list[SchemaDoc] = []
    for entry in raw:
        docs.append(
            SchemaDoc(
                id=entry["id"],
                kind=entry["kind"],
                name=entry["name"],
                description=entry.get("description") or "",
                tags=tuple(entry.get("tags") or []),
                parent_table=entry.get("parent_table"),
                text=_build_text(entry),
            )
        )
    docs.sort(key=lambda d: d.id)
    return docs


def doc_ids(docs: Iterable[SchemaDoc]) -> list[str]:
    return [d.id for d in docs]
