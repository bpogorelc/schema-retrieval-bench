from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from schema_bench.paths import QUESTIONS_PATH

Split = Literal["dev", "held_out"]


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    intent_family: str
    split: Split
    relevant: dict[str, int]


@dataclass(frozen=True)
class QuestionBank:
    label_status: str
    questions: list[Question]

    def by_split(self, split: Split) -> list[Question]:
        return [q for q in self.questions if q.split == split]

    def by_ids(self, ids: list[str]) -> list[Question]:
        by_id = {q.id: q for q in self.questions}
        return [by_id[i] for i in ids if i in by_id]


def load_questions(path: Path | None = None) -> QuestionBank:
    qpath = path or QUESTIONS_PATH
    raw = yaml.safe_load(qpath.read_text(encoding="utf-8"))
    questions: list[Question] = []
    for item in raw["questions"]:
        split = item["split"]
        if split not in ("dev", "held_out"):
            raise ValueError(f"Invalid split for {item['id']}: {split}")
        relevant = {str(k): int(v) for k, v in (item.get("relevant") or {}).items()}
        questions.append(
            Question(
                id=item["id"],
                text=item["text"],
                intent_family=item["intent_family"],
                split=split,  # type: ignore[arg-type]
                relevant=relevant,
            )
        )
    return QuestionBank(
        label_status=str(raw.get("label_status", "unknown")),
        questions=questions,
    )


def validate_intent_family_splits(bank: QuestionBank) -> None:
    """Ensure paraphrases of one intent stay in the same split."""
    family_splits: dict[str, set[str]] = {}
    for q in bank.questions:
        family_splits.setdefault(q.intent_family, set()).add(q.split)
    leaked = {f: s for f, s in family_splits.items() if len(s) > 1}
    if leaked:
        raise ValueError(f"intent_family split leakage: {leaked}")
