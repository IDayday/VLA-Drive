"""Execute the locked action-only SFT methods solely for integration comparisons.

These methods are AST extractions from the release's exact source, independent
of the refactored condition/velocity implementations under test.
"""
from contextlib import contextmanager, nullcontext
import importlib
from pathlib import Path
import types


@contextmanager
def action_only_source(policy, contract):
    framework_scope = dict(
        vars(importlib.import_module("starVLA.model.framework.QwenOFT"))
    )
    framework_scope["nullcontext"] = nullcontext
    visual = Path("tests/flow_grpo/vendor/action_only_visual.py")
    exec(compile(visual.read_text(), str(visual), "exec"), framework_scope)
    framework = Path(contract["framework_oracle"])
    exec(compile(framework.read_text(), str(framework), "exec"), framework_scope)
    head_scope = dict(
        vars(
            importlib.import_module(
                "starVLA.model.modules.action_model.GR00T_ActionHeader"
            )
        )
    )
    head = Path(contract["action_oracle"])
    exec(compile(head.read_text(), str(head), "exec"), head_scope)
    saved = []

    def replace(obj, name, value):
        saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, value)

    for obj, cls in [
        (policy, framework_scope["Qwenvl_OFT"]),
        (policy.action_model, head_scope["FlowmatchingActionHead"]),
    ]:
        for name, method in vars(cls).items():
            if name.startswith("__"):
                continue
            if isinstance(method, staticmethod):
                replace(obj, name, method.__func__)
            elif callable(method):
                replace(obj, name, types.MethodType(method, obj))
    replace(policy, "qwen_visual_frozen", True)
    replace(policy, "_use_named_loss_contract", False)
    try:
        yield policy
    finally:
        for obj, name, existed, value in reversed(saved):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
