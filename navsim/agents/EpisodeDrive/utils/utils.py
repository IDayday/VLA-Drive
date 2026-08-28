
from omegaconf import DictConfig, OmegaConf

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


def build_drivevla_questions(history_trajectory, high_command_one_hot):
    """Build the exact InternVL driving prompts for a batch.

    Keeping this CPU-only helper outside the agent lets DataLoader workers do
    the string formatting ahead of time.  The released implementation built
    the same strings after moving the tensors to CUDA and therefore performed
    24 synchronizing ``Tensor.item()`` calls for every two-sample batch.
    """
    if history_trajectory.ndim == 2:
        history_trajectory = history_trajectory.unsqueeze(0)
    if high_command_one_hot.ndim == 1:
        high_command_one_hot = high_command_one_hot.unsqueeze(0)

    navigation_commands = ["turn left", "go straight", "turn right", "unknown"]
    command_indices = high_command_one_hot.argmax(dim=-1)
    questions = []
    for index in range(history_trajectory.shape[0]):
        history_sample = history_trajectory[index]
        command = navigation_commands[int(command_indices[index].item())]
        history_str = " ".join(
            [
                f"   - t-{3-step}: ({format_number(history_sample[step, 0].item())}, "
                f"{format_number(history_sample[step, 1].item())}, "
                f"{format_number(history_sample[step, 2].item())})"
                for step in range(history_sample.shape[0])
            ]
        )
        prompt = (
            "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{command.upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        questions.append(f"{prompt}{output_requirements}")

    return questions
