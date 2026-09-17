"""Inventory locked official target windows without loading all logs into RAM."""
from pathlib import Path
import json
import pickle
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.loading import file_sha
from starVLA.rl.flow_grpo.reward import build_cache_index


def main():
    base = Path("/mnt/project/DriveDreamer-Policy")
    config = Path(
        "navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    )
    cfg = OmegaConf.load(config)
    prepared = set(json.loads((base / "navtrain_meta.json").read_text()))
    index = build_cache_index(base / "navsim_exp/qds_metric_cache_navtrain")
    logs = base / "navsim_raw/navsim_logs/trainval"
    missing_logs = []
    tokens = []
    missing_image = 0
    requested_tokens = set(cfg.tokens) if cfg.get("tokens") is not None else None
    for log in cfg.log_names:
        path = logs / (log + ".pkl")
        if not path.is_file():
            missing_logs.append(log)
            continue
        frames = pickle.load(path.open("rb"))
        for start in range(0, len(frames), cfg.frame_interval):
            if start + cfg.num_history_frames + cfg.num_future_frames > len(frames):
                continue
            current = frames[start + cfg.num_history_frames - 1]
            if cfg.has_route and not current["roadblock_ids"]:
                continue
            token = current["token"]
            if requested_tokens is not None and token not in requested_tokens:
                continue
            tokens.append(token)
            # Paths in official frame dictionaries are relative to sensor root.
            cams = current.get("cams", {})
            for key in ("CAM_F0", "CAM_L0", "CAM_R0"):
                image = cams.get(key, {}).get("data_path")
                if (
                    image is None
                    or not (base / "navsim_raw/sensor_blobs/trainval" / image).is_file()
                ):
                    missing_image += 1
    result = {
        "official_filter_sha256": file_sha(config),
        "official_filter_logs": len(cfg.log_names),
        "official_filter_tokens": len(requested_tokens)
        if requested_tokens is not None
        else None,
        "missing_raw_logs": missing_logs,
        "available_official_scene_windows": len(tokens),
        "unique_tokens": len(set(tokens)),
        "prepared_meta_count": len(prepared),
        "prepared_outside_target": sorted(prepared - set(tokens)),
        "target_without_prepared_meta": len(set(tokens) - prepared),
        "target_without_metric_cache": len(set(tokens) - index.keys()),
        "target_missing_current_view_files": missing_image,
        "complete_ready_target": not missing_logs
        and not (set(tokens) - prepared)
        and not (set(tokens) - index.keys())
        and not missing_image,
        "source_u_expected_scenes": 103288,
    }
    root = Path("reports/ddp_flow_grpo_paired")
    (root / "target_inventory.json").write_text(json.dumps(result, indent=2))
    (root / "official_available_navtrain_tokens.json").write_text(
        json.dumps(sorted(set(tokens)))
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if not isinstance(v, list)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
