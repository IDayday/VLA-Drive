"""Own-SFT paired analysis, with qualified reuse only for A's exact initialization.

All score loading, safety transitions, bootstrap and cost calculations are the
existing analyzer. No historical prediction/manifest is rewritten for reuse.
"""
import json
from pathlib import Path
from scripts.analysis.simwam_recipe.analyze_evaluation import main
from scripts.analysis.simwam_recipe.evaluation_compatibility import match_protocol, MATCHED_FIELDS
from scripts.analysis.fm_transfer.milestone_controller import BASELINE_RECEIPT_SHA
from starVLA.rl.fm_transfer.contracts import ROOT
from starVLA.rl.simwam_recipe.contracts import file_sha, SFT_SHA256

EVALUATION_SOURCE=ROOT/'source/45d2318d2dec265583860a26f891d8b79bb6068a'


def match_transfer(identity,reference,receipt=None):
    if identity['schema']!='fm_transfer_raw_observation_dual_grid_v1':
        raise ValueError('Not a new-base deployment evaluation')
    if reference['schema']==identity['schema']:
        for key in ('architecture','shift5_time_field','transfer_evaluation_sha256'):
            if identity[key]!=reference[key]:raise ValueError('Unmatched new-base protocol: '+key)
        if identity['new_SFT']['config']['variant']!=reference['new_SFT']['config']['variant']:
            raise ValueError('Within-route gains must use the same architecture lineage')
        if identity['stage']=='RL' and identity['new_SFT']!=reference['new_SFT']:
            raise ValueError('RL must compare with its own exact supervised parent')
        return match_protocol(identity,reference,receipt)
    # The only cross-schema exemption is a real equality-verified initial A
    # baseline. It does not waive source/model checks for any new policy.
    proof_path=ROOT/'qualification/A_DDP_TIME/initial_baseline_reuse.json'
    proof=json.loads(proof_path.read_text())
    if file_sha(proof_path)!=BASELINE_RECEIPT_SHA or proof['status']!='PASS':
        raise ValueError('Changed initial reuse qualification')
    if (identity['stage']!='SFT' or identity['architecture']!='ddp' or
            identity['new_SFT']['config']['variant']!='A_DDP_TIME' or reference['weight_sha256']!=SFT_SHA256):
        raise ValueError('Historical F-SFT is only A initialization, never the new RL parent')
    reuse=proof['reuse'][identity['cohort']];ref_path=Path(reuse['path'])/'manifest.json'
    if file_sha(ref_path)!=reuse['manifest_sha256'] or json.loads(ref_path.read_text())!=reference:
        raise ValueError('Reference is not the authenticated historical initial baseline')
    if (identity['source_sha256']!=file_sha(EVALUATION_SOURCE/'scripts/analysis/simwam_recipe/evaluate_policy.py') or
            identity['transfer_evaluation_sha256']!=file_sha(EVALUATION_SOURCE/'scripts/analysis/fm_transfer/evaluate_policy.py')):
        raise ValueError('A reuse qualification does not cover this evaluation implementation')
    for key in MATCHED_FIELDS:
        if key not in ('schema','actor') and identity[key]!=reference[key]:
            raise ValueError('Changed raw observation/score population: '+key)
    return dict(mode='qualified_exact_initial_A_baseline_reuse',qualification=str(proof_path),
        qualification_sha256=BASELINE_RECEIPT_SHA,all_initial_model_tensors_equal=True,
        actual_raw_observation_dual_grid_smoke_exact=True,
        reference_manifest_sha256=reuse['manifest_sha256'],
        metadata_differences=['schema','actor','source_sha256'],
        not_applicable_to_RL_parent=True,prediction_score_and_population_mutations=False)


if __name__=='__main__':main(protocol_matcher=match_transfer)
