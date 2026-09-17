# Exact optimizer grouping from IDayday/VLA-Drive f9449d55bea6895a7a0bd86d09d7ab85fd353f26.
# Source: starVLA/training/trainer_utils/trainer_tools.py; repository Apache-2.0.


def build_param_lr_groups(model, cfg):
    """
    build multiple param groups based on cfg.trainer.learning_rate.
    support specifying different learning rates for different modules, the rest use base.

    Args:
        vla: nn.Module model object
        cfg: config object, requires cfg.trainer.learning_rate dictionary

    Returns:
        List[Dict]: param_groups that can be used to build optimizer with torch.optim
    """
    lr_cfg = cfg.trainer.learning_rate
    base_lr = lr_cfg.get("base", 0.0001)
    freeze_modules = cfg.trainer.get("freeze_modules", "")
    if not isinstance(freeze_modules, str):
        freeze_modules = ""
    freeze_patterns = [p.strip() for p in freeze_modules.split(",") if p.strip()]
    used_params = set()
    frozen_params = set()
    param_groups = []
    for freeze_path in freeze_patterns:
        module = model
        try:
            for attr in freeze_path.split("."):
                module = getattr(module, attr)
            frozen_params.update((id(p) for p in module.parameters()))
        except AttributeError:
            print(f"⚠️ freeze module path does not exist: {freeze_path}")
            continue
    configured_modules = []
    for (module_name, lr) in lr_cfg.items():
        if module_name == "base" or lr is None:
            continue
        module = model
        try:
            for attr in module_name.split("."):
                module = getattr(module, attr)
            module_params = list(module.parameters())
            configured_modules.append(
                (len(module_params), module_name, lr, module_params)
            )
        except AttributeError:
            print(f"⚠️ learning-rate module path does not exist: {module_name}")
    configured_modules.sort(key=lambda item: (item[0], item[1]))
    for (_, module_name, lr, module_params) in configured_modules:
        params = [
            parameter
            for parameter in module_params
            if parameter.requires_grad
            and id(parameter) not in frozen_params
            and (id(parameter) not in used_params)
        ]
        if params:
            param_groups.append({"params": params, "lr": lr, "name": module_name})
            used_params.update((id(parameter) for parameter in params))
    other_params = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in used_params and (id(p) not in frozen_params)
    ]
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})
    return param_groups
