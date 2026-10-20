"""Verify strict C/F alignment, or only the explicitly recorded exception."""


def validate_cf_batches(model, c_batch, f_batch, assignment, allowed_delta):
    recorded = assignment.get('token_delta', 0)
    if type(recorded) is not int or abs(recorded) > allowed_delta:
        raise ValueError('Unapproved C/F token exception')
    delta = f_batch['input_ids'].shape[1] - c_batch['input_ids'].shape[1]
    if delta != recorded or f_batch['input_ids'].shape[0] != c_batch['input_ids'].shape[0]:
        raise ValueError('C/F multimodal token length differs from recorded delta')
    for kind in ['history', 'action']:
        c_pos = model._find_token_positions(c_batch['input_ids'], model._special_token_ids[kind])
        f_pos = model._find_token_positions(f_batch['input_ids'], model._special_token_ids[kind])
        if c_pos.shape != f_pos.shape or not (f_pos - c_pos).eq(recorded).all().item():
            raise ValueError('C/F insertion positions differ from recorded delta')
    return {'token_delta': recorded, 'strict': recorded == 0}
