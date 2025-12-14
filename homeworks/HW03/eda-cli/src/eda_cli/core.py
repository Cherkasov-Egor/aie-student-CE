import pandas as pd
from eda_cli.core import compute_quality_flags, summarize_dataset, missing_table


def test_has_constant_columns():
    df = pd.DataFrame({
        "A": [1, 1, 1, 1],
        "B": [1, 2, 3, 4],
        "C": ["x", "x", "x", "x"]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_constant_columns"] is True
    assert set(flags["constant_columns_list"]) == {"A", "C"}


def test_has_high_cardinality_categoricals():
    df = pd.DataFrame({
        "id": list(range(95)),
        "normal_cat": ["A", "B"] * 47 + ["C", "D"]
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["has_high_cardinality_categoricals"] is True
    assert "id" in flags["high_cardinality_columns_list"]


def test_quality_score_decreases_with_new_flags():
    df = pd.DataFrame({
        "A": [1, 1, 1, 1],
        "id": list(range(95))
    })
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)

    assert flags["quality_score"] < 1.0
