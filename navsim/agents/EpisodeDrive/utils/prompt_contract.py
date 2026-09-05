"""Versioned actual tokenizer prompt; legacy remains available for replay."""
import hashlib


def resolve_system_message(legacy, version="legacy"):
    if version == "legacy":
        return legacy
    if version != "single_front_v1p1":
        raise ValueError(f"Unknown prompt contract: {version}")
    return legacy.replace(
        "multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history,",
        "one current front-camera image, ego position history, and discrete navigation commands. The visual input contains only the current frame,"
    )


def prompt_sha256(message):
    return hashlib.sha256(message.encode("utf-8")).hexdigest()
