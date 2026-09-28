import numpy as np
import pytest
import torch
from torch import nn
from starVLA.model.modules.vehicle_joint.initialization import (
    initialization_seed, tensor_hash, verify_generic_source, file_sha256,
    add_random_driving_tokens,
)
from starVLA.model.modules.vehicle_joint.targets import make_vehicle_targets


def frame(names, xy, timestamp=0):
    boxes = np.zeros((len(names), 7), dtype=np.float32)
    boxes[:, :2] = xy
    boxes[:, 3:6] = [4., 2., 1.5]
    return {"timestamp": timestamp, "ego2global": np.eye(4), "anns": {
        "gt_boxes": boxes, "gt_names": names, "track_tokens": [str(i) for i in range(len(names))]}}


def test_vehicle_filter_before_capacity_and_same_track():
    current = frame(["pedestrian", "traffic_cone", "vehicle", "vehicle"],
                    [[2, 0], [3, 0], [9, 0], [10, 0]])
    future = frame(["pedestrian", "traffic_cone", "vehicle", "vehicle"],
                   [[2, 0], [3, 0], [10, 0], [10, 0]], 500000)
    target, audit = make_vehicle_targets(current, [future], capacity=2)
    assert target.track_ids == ("2", "3")
    assert target.current_classes.tolist() == [0, 0]
    assert target.future_xy_in_ego_t0[:, 0, 0].tolist() == [10, 10]
    assert target.future_valid_mask[:, 0].all()
    assert not target.future_valid_mask[:, 1:].any()
    assert audit["source_current_objects"] == 4 and audit["source_vehicles"] == 2


def test_empty_and_missing_annotations_are_different():
    empty, _ = make_vehicle_targets(frame([], np.empty((0, 2))), [])
    missing, _ = make_vehicle_targets({"ego2global": np.eye(4), "timestamp": 0}, [])
    assert empty.annotation_valid_mask and not missing.annotation_valid_mask
    assert empty.current_boxes.shape == missing.current_boxes.shape == (0, 8)


def test_new_module_rng_does_not_change_common_parameters():
    torch.manual_seed(42)
    expected = nn.Linear(5, 3)
    torch.manual_seed(42)
    with initialization_seed(1001): nn.Linear(29, 81)
    actual = nn.Linear(5, 3)
    assert tensor_hash(actual.weight) == tensor_hash(expected.weight)
    with initialization_seed(1001): first = nn.Linear(29, 81)
    with initialization_seed(1001): second = nn.Linear(29, 81)
    assert tensor_hash(first.weight) == tensor_hash(second.weight)


def test_generic_loader_rejects_driving_sources_and_tampering(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"public generic fixture")
    record = {"repository": "Qwen/Qwen3-VL-2B-Instruct", "revision": "a" * 40,
              "files": {weight.name: file_sha256(weight)}}
    assert verify_generic_source(tmp_path, record)["revision"] == "a" * 40
    with pytest.raises(ValueError, match="allowed generic"):
        verify_generic_source(tmp_path, dict(record, repository="local/driving-policy"))
    weight.write_bytes(b"different driving weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_generic_source(tmp_path, record)


def test_spare_vocabulary_rows_randomized_native_rows_preserved():
    from types import SimpleNamespace
    class Tokenizer:
        def __init__(self): self.vocab = {"native": 0, "image": 1}
        def get_vocab(self): return self.vocab
        def add_special_tokens(self, values, **kwargs):
            for token in values["additional_special_tokens"]: self.vocab[token] = len(self.vocab)
        def convert_tokens_to_ids(self, values): return [self.vocab[t] for t in values]
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(10, 4)
            self.config = SimpleNamespace(text_config=SimpleNamespace(initializer_range=.02))
        def get_input_embeddings(self): return self.embed
        def get_output_embeddings(self): return self.embed
    model, tokenizer = Model(), Tokenizer()
    original = model.embed.weight.detach().clone()
    record = add_random_driving_tokens(model, tokenizer, ["driving"], 71)
    assert record["tokens"] == {"driving": 2}
    assert torch.equal(model.embed.weight[:2], original[:2])
    assert not torch.equal(model.embed.weight[2], original[2])
    with pytest.raises(ValueError, match="absent"):
        add_random_driving_tokens(model, tokenizer, ["driving"], 71)
