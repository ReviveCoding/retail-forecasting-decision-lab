"""Public dataset paths should not depend on a developer's machine."""

from pathlib import Path

from src.config import load_pipeline_config


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pipeline.yaml"


def test_public_default_path_is_repository_relative(monkeypatch):
    monkeypatch.delenv("M5_DATA_PATH", raising=False)
    cfg = load_pipeline_config(DEFAULT_CONFIG)
    assert cfg["data"]["m5_path"] == "data/raw/m5-forecasting-accuracy.zip"
    for field in (
        "favorita_path",
        "rossmann_path",
        "olist_path",
        "instacart_path",
        "online_retail_ii_path",
    ):
        assert cfg["extensions"][field].startswith("data/external/")


def test_dataset_environment_override(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "m5-extracted"
    monkeypatch.setenv("M5_DATA_PATH", str(dataset_dir))
    cfg = load_pipeline_config(DEFAULT_CONFIG)
    assert cfg["data"]["m5_path"] == str(dataset_dir)


def test_custom_yaml_path_wins_without_environment_override(monkeypatch, tmp_path):
    monkeypatch.delenv("M5_DATA_PATH", raising=False)
    custom = tmp_path / "custom.yaml"
    custom.write_text("data:\n  m5_path: data/custom/m5.zip\n", encoding="utf-8")
    cfg = load_pipeline_config(custom)
    assert cfg["data"]["m5_path"] == "data/custom/m5.zip"
