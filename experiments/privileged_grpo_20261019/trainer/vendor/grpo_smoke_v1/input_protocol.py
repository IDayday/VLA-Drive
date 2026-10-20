"""Action-only runtime inputs; original checkpoint/tokenizer stay unchanged."""
from contextlib import contextmanager
INPUT_PROTOCOL_VERSION = 'action-only-suffix-v3'


def suffix(model):
    datasets = model.config.datasets
    if (model.w_depth or datasets.video_data.load_2d_data
            or datasets.gs_data.load_3d_data or datasets.reward_data.load_reward_data):
        raise ValueError('Action-only suffix requires all auxiliary branches disabled')
    if len(model.act_query_tokens) != 8:
        raise ValueError('Expected eight action queries')
    return ' ' + model.robot_history_token + ''.join(model.act_query_tokens)


def validate_inputs(model, batch):
    """Check actual multimodal IDs, including accidental prompt token injection."""
    ids = batch['input_ids']
    mask = batch['attention_mask'].bool()
    if ids.ndim != 2 or mask.shape != ids.shape:
        raise ValueError('Invalid multimodal input shape')
    special = model._special_token_ids
    expected = tuple(special['history']) + tuple(special['action'])
    if len(special['history']) != 1 or len(special['action']) != 8 or len(set(expected)) != 9:
        raise ValueError('Expected distinct history and eight action token IDs')
    inactive = ids.new_tensor(tuple(special['rgb']) + tuple(special['gs']) + tuple(special['reward']))
    if bool(((ids[:, :, None] == inactive) & mask[:, :, None]).any()):
        raise ValueError('Inactive RGB/GS/reward query token in input')
    for row, active in zip(ids, mask):
        sequence = row[active]
        positions = []
        for token_id in expected:
            found = (sequence == token_id).nonzero().flatten()
            if found.numel() != 1:
                raise ValueError('History/action token must appear exactly once')
            positions.append(int(found[0]))
        if positions != list(range(positions[0], positions[0] + len(expected))):
            raise ValueError('History/action suffix must remain contiguous and ordered')


def validate_protocol(cfg):
    if cfg.get('input_protocol_version') != INPUT_PROTOCOL_VERSION:
        raise ValueError('Input protocol mismatch; use a new config/run for ' + INPUT_PROTOCOL_VERSION)


@contextmanager
def native_clean_suffix(model):
    """Adapt native replay input without editing the archived model source/weights."""
    suffix(model)  # Reject enabled auxiliary heads before changing token lists.
    names = ('rgb_query_tokens', 'gs_query_tokens', 'reward_query_tokens')
    original = {name: getattr(model, name) for name in names}
    try:
        for name in names:
            setattr(model, name, [])
        yield
    finally:
        for name, value in original.items():
            setattr(model, name, value)
