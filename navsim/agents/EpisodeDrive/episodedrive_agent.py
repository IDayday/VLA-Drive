"""EpisodeDrive agent entry point for NAVSIM."""

from .drivevla_base_agent import DriveVLABaseAgent

class EpisodeDriveAgent(DriveVLABaseAgent):
    """DriveVLA-M0 base model agent exposed with the paper-facing name."""
    pass

__all__ = ["DriveVLABaseAgent", "EpisodeDriveAgent"]
