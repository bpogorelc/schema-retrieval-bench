from schema_bench.labels import load_questions, validate_intent_family_splits


def test_intent_family_no_leakage():
    bank = load_questions()
    validate_intent_family_splits(bank)


def test_draft_label_status():
    bank = load_questions()
    assert bank.label_status == "draft"


def test_question_count_and_splits():
    bank = load_questions()
    assert len(bank.questions) == 30
    assert len(bank.by_split("dev")) >= 15
    assert len(bank.by_split("held_out")) >= 8


def test_catalog_size():
    from schema_bench.catalog import load_catalog

    docs = load_catalog()
    assert 40 <= len(docs) <= 60
