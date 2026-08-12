"""
Framework factory utilities.
Automatically builds registered framework implementations
based on configuration.

Each framework module (e.g., M1.py, QwenFast.py) should register itself:
    from starVLA.model.framework.framework_registry import FRAMEWORK_REGISTRY

    @FRAMEWORK_REGISTRY.register("InternVLA-M1")
    def build_model_framework(config):
        return InternVLA_M1(config=config)
"""

import importlib
from starVLA.model.tools import FRAMEWORK_REGISTRY

_LAZY_FRAMEWORK_MODULES = {
    "InternVLA-M1": "M1",
    "Qwen-Dual": "QwenDual",
    "QwenPI": "QwenPI",
    "QwenGR00T": "QwenGR00T",
    "QwenOFT_s2": "QwenOFT_s2",
}
        
def build_framework(cfg, accelerator=None):
    """
    Build a framework model from config.
    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.name: Identifier string (e.g. "InternVLA-M1")
    Returns:
        nn.Module: Instantiated framework model.
    """

    if not hasattr(cfg.framework, "name"): 
        cfg.framework.name = cfg.framework.framework_py  # Backward compatibility for legacy config yaml
        
    if cfg.framework.name == "QwenOFT":
        from starVLA.model.framework.QwenOFT import Qwenvl_OFT
        return Qwenvl_OFT(cfg, accelerator)
    elif cfg.framework.name == "QwenOFT_VGGT":
        from starVLA.model.framework.QwenOFT_VGGT import Qwenvl_OFT_VGGT
        return Qwenvl_OFT_VGGT(cfg, accelerator)
    elif cfg.framework.name == "QwenFast":
        from starVLA.model.framework.QwenFast import Qwenvl_Fast
        return Qwenvl_Fast(cfg)
    elif cfg.framework.name in ("QWenGROOT", "QwenGR00T"):
        from starVLA.model.framework.QwenGR00T import Qwen_GR00T
        return Qwen_GR00T(cfg)
    elif cfg.framework.name == "QwenPI":
        from starVLA.model.framework.QwenPI import Qwen_PI
        return Qwen_PI(cfg)

    elif cfg.framework.name == "QwenVision":
        from starVLA.model.framework.QWenVision import Qwenvl_Vision
        return Qwenvl_Vision(cfg)

    
    # auto detect from registry
    framework_id = cfg.framework.name
    module_name = _LAZY_FRAMEWORK_MODULES.get(framework_id)
    if module_name is not None and framework_id not in FRAMEWORK_REGISTRY._registry:
        importlib.import_module(f"{__name__}.{module_name}")
    if framework_id not in FRAMEWORK_REGISTRY._registry:
        raise NotImplementedError(f"Framework {cfg.framework.name} is not implemented.")
    
    MODLE_CLASS = FRAMEWORK_REGISTRY[framework_id]
    return MODLE_CLASS(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]
