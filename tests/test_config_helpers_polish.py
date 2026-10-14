"""Regression tests for nested and legacy configuration helpers."""

from src.config import _legacy_to_nested, _merge_dict


def test_merge_preserves_unmodified_defaults_and_nested_values():
    defaults = {"model": {"seed": 7, "epochs": 2}, "data": {"rows": 100}}
    overrides = {"model": {"epochs": 4}, "data": {"rows": 200}}
    result = _merge_dict(defaults, overrides)
    assert result == {"model": {"seed": 7, "epochs": 4}, "data": {"rows": 200}}
    assert defaults == {"model": {"seed": 7, "epochs": 2}, "data": {"rows": 100}}


def test_legacy_flat_config_retains_cross_section_seed():
    config = _legacy_to_nested({"random_seed": 11, "n_items": 20})
    assert config["data"]["n_items"] == 20
    assert config["data"]["random_seed"] == 11
    assert config["model"]["random_seed"] == 11
    assert config["experiment"]["random_seed"] == 11


def test_nested_config_is_not_rewritten():
    config = {"data": {"n_items": 4}}
    assert _legacy_to_nested(config) is config
