"""Materialize audited local configs without changing downloaded source/weights."""
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf

SOURCE_SHA = "f9449d55bea6895a7a0bd86d09d7ab85fd353f26"
SOURCE = Path("artifacts/action-only-source")
ASSETS = Path("artifacts/action-only-checkpoints-v1")
MAPPING = {
    "frozen_visual": (
        "frozen-visual-step100000-config.yaml",
        "action-only-frozen-visual-step100000-pdms88.79.pt",
        "9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4",
    ),
    "unfrozen_visual": (
        "unfrozen-visual-step120000-run-config.yaml",
        "action-only-unfrozen-visual-step120000-pdms89.575.pt",
        "72e626e152a357ec9c478c4b253500599301b38b87758e0aebfa486ff3be21cc",
    ),
}


def main():
    from fetch_action_only_source import fetch

    fetch()
    for variant, (config_name, weight_name, sha) in MAPPING.items():
        source_config = (
            SOURCE / "checkpoint_code/action-only-checkpoints-v1/configs" / config_name
        )
        original = OmegaConf.load(source_config)
        cfg = OmegaConf.load("configs/flow_grpo/full_sft.yaml")
        checkpoint = (ASSETS / variant).resolve()
        checkpoint.mkdir(parents=True, exist_ok=True)
        raw = (ASSETS / weight_name).resolve()
        link = checkpoint / "pytorch_model.pt"
        if not link.is_symlink():
            link.symlink_to(raw)
        effective = OmegaConf.create(OmegaConf.to_container(original, resolve=True))
        if variant == "frozen_visual":
            # The saved YAML has an empty field; the archived launcher and release
            # mapping explicitly record these two freezes. Preserve both sources.
            effective.trainer.freeze_modules = (
                "qwen_vl_interface.model.visual,qwen_vl_interface.model.lm_head"
            )
        else:
            effective.framework.retain_inactive_agent_dino_head = True
        OmegaConf.save(effective, checkpoint / "config.yaml")
        cfg.sft_checkpoint = str(checkpoint)
        cfg.rl_freeze_modules = ["qwen_vl_interface.model.visual"]
        cfg.runtime.noise_seed_schedule = "global_scene_v1"
        cfg.checkpoint_contract = dict(
            origin="action_only_release",
            variant=variant,
            code_sha=SOURCE_SHA,
            sha256=sha,
            sft_feature_output="normalized",
            policy_feature_output="normalized",
            source_config=str(source_config),
            source_config_sha256=hashlib.sha256(source_config.read_bytes()).hexdigest(),
            framework_oracle=f"tests/flow_grpo/vendor/{variant}_framework.py",
            action_oracle="tests/flow_grpo/vendor/action_only_head.py",
        )
        OmegaConf.save(cfg, f"configs/flow_grpo/action_only_{variant}.yaml")
        (checkpoint / "source_provenance.json").write_text(
            json.dumps(
                dict(
                    source_sha=SOURCE_SHA,
                    source_config=str(source_config),
                    source_freeze_field=original.trainer.freeze_modules,
                    effective_sft_freezes=effective.trainer.freeze_modules,
                    rl_additional_freezes=list(cfg.rl_freeze_modules),
                    frozen_contract_evidence="archived 8-train.sh and ACTION_ONLY_CHECKPOINTS.md/release mapping",
                    weight_sha256=sha,
                ),
                indent=2,
            )
        )
        print(variant, checkpoint, "download_complete=", raw.is_file())


if __name__ == "__main__":
    main()
