import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.prepare_merged_vqa_vlm_init import classify_vqa_checkpoint


def test_vqa_checkpoint_kind_detection(tmp_path: Path):
    dense = tmp_path / "dense"
    dense.mkdir()
    save_file({"weight": torch.ones(1)}, str(dense / "model.safetensors"))
    assert classify_vqa_checkpoint(dense) == "dense"

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert classify_vqa_checkpoint(adapter) == "peft"


def test_vqa_checkpoint_rejects_agent_like_ambiguous_layout(tmp_path: Path):
    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    (ambiguous / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": torch.ones(1)}, str(ambiguous / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Ambiguous"):
        classify_vqa_checkpoint(ambiguous)


def test_vqa_checkpoint_requires_weights_or_adapter(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="neither a dense VLM nor a PEFT"):
        classify_vqa_checkpoint(empty)
