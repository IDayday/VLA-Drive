"""Resolve the user's framework-only correction and audit actual local sources."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import platform
import subprocess
from .contracts import require
from .io import atomic_json, atomic_bytes, file_hash, digest, read_json, BlockedError
from .query_base import source_trajectory_contract
from .scoring.reference import python_source_hash, COMPONENTS


def verify_existing_contract(previous, current, contract_path, output, *, dependency_hashes):
    """Retain historical identity when only unrelated post-training code changed.

    Never rewrite the old contract/hash. All original source, VLM, protected
    model file hashes, architecture, inputs, losses and metric fields must match.
    """
    excluded = {'query_checkpoint', 'query_checkpoint_hash', 'binding_status', 'workspace', 'framework_contract_hash'}
    require(previous['framework_contract_hash'] == digest({k:v for k,v in previous.items() if k not in excluded}),
            'historical framework contract checksum invalid')
    if previous['framework_contract_hash'] == current['framework_contract_hash']:
        return previous
    semantic_excluded = excluded | {'iqe_code_hash'}
    require(digest({k:v for k,v in previous.items() if k not in semantic_excluded}) ==
            digest({k:v for k,v in current.items() if k not in semantic_excluded}),
            'preflight model/input/source/metric contract changed; explicit model compatibility migration required')
    receipt = {'kind':'POSTTRAIN_CODE_ONLY_COMPATIBILITY', 'historical_contract':str(Path(contract_path).resolve()),
               'historical_contract_checksum':file_hash(contract_path), 'framework_contract_hash':previous['framework_contract_hash'],
               'original_iqe_code_hash':previous['iqe_code_hash'], 'current_iqe_code_hash':current['iqe_code_hash'],
               'protected_model_hashes':current['query_model_code_hashes'], 'current_workspace':current['workspace'],
               'additional_base_dependency_hashes':dependency_hashes,
               'semantic_contract_unchanged':True, 'historical_contract_rewritten':False,
               'checkpoint_load_rule':'unchanged strict state_dict plus protected-file hash validation'}
    path=Path(output)/'preflight_compatibility'/digest(receipt)/'COMPATIBILITY.json'
    atomic_json(path,receipt,immutable=True)
    atomic_json(Path(output)/'PREFLIGHT_COMPATIBILITY.json',{'path':str(path.resolve()),'checksum':file_hash(path)})
    return previous


def audited_base_dependencies(repo):
    # The closeout starts from this user-pinned audit anchor. Contracts contains
    # physical conversion and io contains strict weight loading/hash utilities;
    # neither may hide behind a change to the aggregate post-training hash.
    anchor='924b003ad84ffc2d51e1cb4f5e140663f3280152'
    result={}
    for name in ('iqe/contracts.py','iqe/io.py'):
        expected=__import__('hashlib').sha256(subprocess.check_output(['git','show',anchor+':'+name],cwd=repo)).hexdigest()
        require(file_hash(Path(repo)/name)==expected, 'Base dependency changed; explicit model compatibility migration required: '+name)
        result[name]=expected
    return {'audit_anchor':anchor,'files':result}


def audit(config, repo):
    repo = Path(repo).resolve()
    s0 = config["s0"]
    source = Path(s0["source_root"])
    require(source.is_dir() and Path(s0["source_config"]).is_file(), "real source/config unavailable")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    require(commit == s0["source_commit"], "S0 source commit mismatch")
    original = read_json(s0["source_config"])
    require(original["framework"]["name"] == "DDPActionVideoForesight" and original["foresight"]["future_action_condition"] == "none", "wrong original S0 framework")
    require(s0["learned_driving_checkpoint"] is None, "learned driving checkpoint forbidden by user's correction")
    vlm = Path(original["framework"]["qwenvl"]["base_vlm"])
    require(vlm.is_dir(), "actual generic VLM missing")
    cfg = read_json(vlm / "config.json")
    hidden = cfg.get("text_config", cfg)["hidden_size"]
    require(hidden == original["framework"]["qwenvl"]["vl_hidden_dim"], "real VLM hidden width mismatch")
    files = ("starVLA/model/framework/QwenOFT.py", "starVLA/model/framework/DDPForesight.py",
             "starVLA/model/framework/ddp_full_foresight.py", "starVLA/model/framework/ddp_action_video_foresight.py",
             "starVLA/dataloader/foresight_dataset.py", "tools/foresight/train_student.py", "tools/foresight/score_pdms.py",
             "tools/local_interaction_mask_v2/score_async.py", "tools/local_interaction_mask_v2/prepare_metric_cache.py")
    source_hashes = {f: file_hash(source / f) for f in files}
    architecture = {k: config["query_base"][k] for k in ("width", "ffn", "layers", "heads", "scene_tokens", "compressor_layers",
                   "compressor_heads", "dropout", "drop_path", "dt", "prev_weight")}
    trajectory = asdict(source_trajectory_contract(original["framework"]["action_model"]["action_horizon"], architecture["dt"]))
    tokenizer = {str(p.relative_to(vlm)): file_hash(p) for p in sorted(vlm.glob("*token*")) if p.is_file()}
    generic_shards = {p.name: file_hash(p) for p in sorted(vlm.glob("*.safetensors"))}
    head_files = {str(p.relative_to(repo)): file_hash(p) for p in (repo / "iqe").rglob("*.py")}
    reused_files = {str(p.relative_to(repo)):file_hash(p) for p in (repo/'navsim/agents/EpisodeDrive').rglob('*.py')}
    registration = Path(s0['source_config']).with_name('formal_S0_seed42_full100k_v2.json')
    base = {"schema_version": 1, "binding_status": "FRAMEWORK_BOUND_QUERY_UNTRAINED",
            "origin": s0["origin"], "source_root": str(source), "source_commit": commit,
            "source_config": original, "source_config_hash": file_hash(s0["source_config"]),
            "source_hashes": source_hashes, "query_architecture": architecture, "trajectory": trajectory,
            "reused_M0_module_hashes": reused_files, "original_registration":read_json(registration),
            "query_model_code_hashes": {name:head_files[name] for name in ('iqe/query_base.py','iqe/expert.py','iqe/losses.py','iqe/s0_adapter.py')},
            "learned_driving_weights_loaded": False, "generic_vlm": str(vlm), "generic_vlm_weights_hashes": generic_shards,
            "VLM_hidden_dim": hidden, "tokenizer_hash": digest(tokenizer),
            "prompt_hash": source_hashes["starVLA/dataloader/foresight_dataset.py"],
            "transform_hash": source_hashes["starVLA/dataloader/foresight_dataset.py"],
            "normalizer_hash": digest(trajectory), "camera_time_hash": digest({"views": ["CAM_F0", "CAM_L0", "CAM_R0"], "time": 0}),
            "scene_memory_shape": [None, architecture["scene_tokens"], architecture["width"]],
            "memory_mask": "all Q-Former scene tokens valid; True means valid",
            "input": {"views": ["CAM_F0", "CAM_L0", "CAM_R0"], "history_images": 0,
                "image_transform": "original center 16:9 crop; Lanczos1024x576; original Qwen processor",
                "ego_state_order": ["normalized_previous_to_current_x", "normalized_previous_to_current_y", "sin_delta_yaw", "cos_delta_yaw"],
                "ego_state_shape": [None, 1, 4], "physical_units": "metres, radians, seconds",
                "original_input_tokens_retained": {"history": 1, "foresight": original["foresight"]["num_queries"], "action": original["act_tok"]},
                "incremental_expert_tokens_added_to_VLM": 0},
            "action": {"query_count": 1, "self_attention": "singleton S0 block retained", "cross_expert_attention": False,
                "heads": architecture["layers"] + 1, "final_tensor": "expert.intermediates[-1]", "hidden_scoring_or_argmax": False,
                "loss": "Query IL: raw channel L1 sum, time mean, recursive inherited M0 prev_weight; not flow matching",
                "loss_change_authorized": "2026-10-09 user: S0 originally DiT flow matching; change to Query; framework not weights",
                "retained_auxiliary_losses": ["current_dino", "future_clip", "interaction"],
                "auxiliary_only_during_query_base_training": True},
            "metric": {"protocol": config["metric"]["protocol"], "root": config["metric"]["navsim_root"],
                "python_source_hash": python_source_hash(Path(config["metric"]["navsim_root"]) / "navsim"),
                "config_hash": digest({k: config["metric"][k] for k in ("proposal_sampling", "scorer")}),
                "reference": "metric_cache.trajectory", "traffic": "official nonreactive recorded observation",
                "components": list(COMPONENTS), "TLC": "absent; no fabricated labels", "train_step_compute_score": False},
            "query_checkpoint": None, "query_checkpoint_hash": None, "iqe_code_hash": digest(head_files),
            "original_run_command": None,
            "original_run_command_evidence": "Exact shell command not found in registration; entry=tools.foresight.train_student, resolved config and complete registration recorded. No invented command.",
            "run_record": "/mnt/project/action-video-foresight-artifacts/20261002/registrations/formal_S0_seed42_full100k_v2.json",
            "training_exposure": {"QueryS0": "not trained yet", "learned_DiT_S0_exposure": "irrelevant: weights not loaded"},
            "dropout": architecture["dropout"], "EMA": "none", "precision": config["model"]["precision"]}
    base["framework_contract_hash"] = digest({k: v for k, v in base.items() if k not in {"query_checkpoint", "query_checkpoint_hash", "binding_status"}})
    base["workspace"] = {"repo": str(repo), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dirty_patch_hash": __import__("hashlib").sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=repo)).hexdigest()}
    path = Path(s0["contract"])
    if path.exists():
        previous = read_json(path)
        return verify_existing_contract(previous,base,path,config['output_root'],dependency_hashes=audited_base_dependencies(repo))
    atomic_json(path, base, immutable=True)
    text = "# S0 contract\n\nFramework reuse only. No learned DiT/S0 driving weights are loaded.\n\n" + \
        "The original Qwen3-VL2B input pipeline is retained; the flow-matching head is replaced by a singleton Query Decoder. " + \
        "The new Query S0 requires its own GT IL base training before incremental expert cloning. " + \
        "Original auxiliary supervision remains during base training and is excluded from deployment.\n\n" + \
        "Source and all actual dimensions, normalization constants, file hashes, metric and input identities are in S0_CONTRACT.json. " + \
        "There is no output-equivalence claim between the DiT head and the newly initialized Query head.\n\n" + \
        f"Framework source: `{commit}`. VLM hidden width: {hidden}. Scene memory:{architecture['scene_tokens']}x{architecture['width']}. " + \
        f"Raw trajectory:{trajectory['horizon']}x4 normalized XY/sin/cos; physical:{trajectory['horizon']}x3 rear-axle relative; dt={trajectory['dt']}s.\n"
    atomic_bytes(path.with_suffix(".md"), text.encode())
    return base
