import pandas as pd
import pytest

from src.data.validate_schema import validate_panel_schema


@pytest.mark.parametrize("reverse", [False, True])
def test_gap_count_keeps_group_boundaries_duplicate_dates_and_nat(reverse):
    frame = pd.DataFrame(
        {
            "item_store_id": ["b", "a", "b", "a", "b", None],
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-02",
                    None,
                    "2026-01-20",
                ]
            ),
        }
    )
    if reverse:
        frame = frame.iloc[::-1]
    result = validate_panel_schema(frame)
    assert result["date_gap_count"] == 1
    assert result["n_series"] == 2
    assert result["null_counts_required"]["date"] == 1
    assert result["null_counts_required"]["item_store_id"] == 1


def test_duplicate_index_does_not_merge_series():
    frame = pd.DataFrame(
        {
            "item_store_id": ["a", "b", "a", "b"],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-04", "2026-01-02"]
            ),
        },
        index=[0, 0, 0, 0],
    )
    assert validate_panel_schema(frame)["date_gap_count"] == 1
