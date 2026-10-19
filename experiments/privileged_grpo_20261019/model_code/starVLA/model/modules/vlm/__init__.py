


def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    try:
        if config.vit_pre:
            from .QWen3 import _QWen3_Vision
            return _QWen3_Vision(config)
    except:
        pass

    if "Qwen2.5-VL" in vlm_name:
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")



