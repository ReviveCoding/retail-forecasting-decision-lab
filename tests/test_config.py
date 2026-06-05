from src.config import load_pipeline_config


def test_pipeline_config_has_nested_sections():
    cfg = load_pipeline_config('configs/pipeline.yaml')
    assert 'data' in cfg
    assert 'model' in cfg
    assert 'inventory' in cfg
    assert 'release_gate' in cfg
