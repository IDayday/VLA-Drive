"""Full-chain ratio observations without redundant intermediate inference.

A subsequent inner epoch evaluates the same chain at exactly the preceding
updated policy. Its pre-update ratios can therefore report that policy's probe.
This does not reuse condition features, gradients, losses, or old log-probs.
"""


def defer_probe(enabled, inner, inner_epochs, update, maximum):
    if not 0 <= inner < inner_epochs or not 0 < update <= maximum:
        raise ValueError("invalid inner-epoch/update boundary")
    # Always observe the final update of this invocation, even a partial batch.
    return enabled and inner + 1 < inner_epochs and update < maximum


def decorate_probe_row(row, enabled, deferred):
    if not enabled:
        return row
    row['post_update_probe_status'] = 'DEFERRED_TO_NEXT_INNER_FORWARD' if deferred else 'COMPLETE'
    row['post_update_probe_source'] = 'next_inner_current_policy_forward' if deferred else 'explicit_no_grad_forward'
    if deferred:
        if 'post_update_probe_ratio_count' in row:
            raise ValueError('deferred observation must not pretend a probe was already measured')
    elif not row.get('post_update_probe_ratio_count', 0):
        raise ValueError('missing explicit final-boundary probe')
    if row['inner_epoch'] > 0:
        prefix = 'pre_update_ratio'
        suffixes = ('min','max','mean','clip_fraction','count','quantiles')
        observation = {'post_update_probe_ratio_'+suffix: row[prefix+'_'+suffix] for suffix in suffixes}
        if observation['post_update_probe_ratio_count'] <= 0:
            raise ValueError('cannot reuse an empty current-policy observation')
        row['previous_update_probe'] = {
            'policy_update': row['update'] - 1,
            'policy_version': row['policy_version'],
            'source': 'pre_update_ratios_on_the_same_complete_behavior_chain',
            'candidate_scope': 'all candidates and valid transitions; no subsampling',
            'metrics': observation,
        }
    return row
