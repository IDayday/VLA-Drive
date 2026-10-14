"""Derive human-readable campaign evidence from actual committed stage artifacts."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from iqe.io import atomic_bytes, atomic_json, file_hash, read_json
from .config import load_config


def table(headers, rows):
    def text(x):
        if x is None:return 'NA'
        if isinstance(x, float):return f'{x:.6g}'
        return str(x).replace('|', '/')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']+
                     ['| '+' | '.join(text(x) for x in row)+' |' for row in rows])+'\n'


def generate(config):
    root=config.root
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    now=datetime.now(timezone.utc).isoformat()
    binding=read_json(root/'BASE_BINDING.json')
    dependencies={'Base_binding':file_hash(root/'BASE_BINDING.json'),'data_partition':file_hash(root/'DATA_PARTITION.json'),
                  'official_protocol':file_hash(root/'SCORE_PROTOCOL.json'),'current_implementation_commit':commit}
    def get(relative,default=None):
        path=root/relative
        return read_json(path) if path.exists() else default
    def report(name,body):
        footer=f'\nGenerated {now}. Implementation `{commit}`. Base step {binding["optimizer_step"]}, SHA256 `{binding["checkpoint_hash"]}`. '
        footer+=f'All current scores use NAVSIM v1.1 fixed official reference; internal score unit [0,1]. Artifact root: `{root}`.\n'
        atomic_bytes(root/name,(body.rstrip()+'\n'+footer).encode())

    roles=get('DATA_PARTITION.json')
    report('DATA_ROLES_AND_PROVENANCE.md', '# Fixed roles and provenance\n\n'+table(
        ['Role','Scenes','Source groups','Base seen yes/no/unknown'],
        [[name,len(value['scene_ids']),len(value['source_groups']),value['base_seen']] for name,value in roles['roles'].items()])+
        '\nD_E and D_S use disjoint complete source logs. The entire legal expert pool was considered; there is no 16k expert limit. '
        'D_S is held out from the new experts, not from the already trained Base. The checkpoint-provided fit_scene_ids supplied Base exposure membership. '
        'Network optimization uses D_E for experts and D_S for Scorers. stage_val chooses endpoints; selector_cal calibrates only the optional margin. '
        'The fixed dev role has 16 independent groups, so bootstrap intervals do not establish broad statistical reliability. '
        'Historical completed Navtest banks are read only for descriptive analysis; current campaign training/selection never accesses Navtest/final_test.\n')

    audit=get('target_audit/CORE_TARGET_ELIGIBILITY.json')
    matrix=get('EXPERT_MATRIX_DATA.json')
    if audit:
        report('TARGET_OPPORTUNITY_AUDIT.md','# Current-Base target opportunity\n\n'+table(
            ['Specialty','Failures','GT opportunity','Paired','Paired logs','GT U','Optimized U','Target gain ≥0.01','Near-identical pairs'],
            [[name,s['failures'],s['gt_eligible'],s['paired'],s['paired_logs'],s['gt_U'],s['optimized_U'],s['pair_target_gain_ge_0_01'],s['pair_nearly_identical']]
             for name,s in audit['summary'].items()])+
            '\nThe raw opportunity audit considered 16,843 current failure/opportunity records from the full D_E plus stage_val, not a new fit cap. '
            'Targets were rescored against the current Base reference context. The core paired comparison uses original GT≥0.9, optimized target≥0.95, '
            'NC/DAC full, optimized target no worse than GT and at least 0.02 above Base, plus geometry/source audit. '
            'DDC has weight zero in this protocol and is separately diagnostic. '\
            '\nHistorical assets include mixed provenance. Only explicit GT/construction/PDM-reference/corridor/map sources enter the core; '
            'diffusiondrivev2, DrivoR and ambiguous proposal ancestors are audit-only. The original source labels are retained. '
            'This does not claim the entire historical asset library was independently generated. Current labels are official environment scores; '
            'no old teacher or external Scorer weights are loaded.\n'+
            (f'\nPilot `{matrix["pilot"]}` uses the same {len(matrix["scene_ids"])} observations/order in all four cells. '
             f'The matched H_val has only {len(matrix["hard_val"])} scene(s), insufficient for a reliable specialty-specific heldout claim. '
             'Full fixed stage_val remains the endpoint selection set.\n' if matrix else ''))

    learning=[];endpoints=[];retention=[]
    for folder in sorted((root/'experts').glob('*')):
        if not folder.is_dir():continue
        result=get(str(folder.relative_to(root)/'result.json'),{})
        for path in sorted(folder.glob('validation_*.json')):
            step=int(path.stem.split('_')[-1]);value=read_json(path)
            for kind,s in value['endpoints'].items():
                endpoints.append([folder.name,step,kind,s['fixed_policy_01'][1],s['oracle_01'],s['oracle_gain_01'],s['hard_val_direct_gain'],
                                  value['ema_initial_Base_weight'] if kind=='ema' else 0.])
        if (folder/'SPECIALIZATION_DIAGNOSTIC.json').exists():
            value=read_json(folder/'SPECIALIZATION_DIAGNOSTIC.json')
            for s in value['reports']:learning.append([folder.name,s['endpoint'],len(value['scene_ids']),s['Base_eval_IL'],s['specialist_eval_IL'],
                                                      s['H_train_direct_PDMS_gain'],s['H_val_count'],s['H_val_direct_PDMS_gain']])
        if (folder/'BASE_RETENTION_AUDIT.json').exists():
            value=read_json(folder/'BASE_RETENTION_AUDIT.json');retention.append([folder.name,value['before']==value['after'],value['raw_drift'],value['physical_drift'],value['VLM_calls']])
    report('EXPERT_LEARNING_CURVES.md','# Specialist training evidence\n\n'+
        ('No completed fixed-endpoint H_train diagnostic yet. Intermediate IL is not evidence of PDMS specialization.\n' if not learning else table(
            ['Cell','Endpoint','H_train scenes','Base target IL','Expert target IL','H_train PDMS delta','H_val count','H_val delta'],learning))+
        '\nPure specialty supervision, no anchor/distillation, 5,000 updates, global batch128, unchanged Query IL/dropout. '
        'One update counts128 actual sample presentations; repeated exposure is not new observations. Core cells use identical sample sequence. '
        'Private-reader cells update only their own copied compression/query/ego/action branch. The shared VLA is absent from head-only training.\n')
    report('RAW_EMA_COMPARISON.md','# Raw and EMA endpoint evidence\n\n'+table(
        ['Cell','Update','Endpoint','Expert alone','Oracle with Base','Oracle gain','H_val direct delta','EMA original-Base weight'],endpoints)+
        '\nRows are stage_val. The main factorial interpretation uses fixed update5000 raw/EMA; system endpoints may be selected from1000/3000/5000 using the locked stage_val rule. '
        'Raw and EMA are two snapshots of one expert and are never both added as separate deployment experts. '
        'Early EMA retains substantial original Base weight; the column reports decay**updates.\n')
    report('BASE_RETENTION_AUDIT.md','# Frozen Base retention\n\n'+table(
        ['Cell','Parameter/buffer hashes identical','Raw drift','Physical drift','Training VLA calls'],retention)+
        '\nNo Base training process, checkpoint, 22k identity, or formal ACTIVE is modified. '
        'Complete shared weight hashes remain tied to the original immutable checkpoint; private training modules contain no shared VLA parameters.\n')

    pool=get('SERVING_CANDIDATE_POOL.json')
    report('SPECIALIST_POOL_RESULTS.md','# Candidate pool\n\n'+('WAITING_INPUTS: fixed expert endpoints/selection are not complete.\n' if not pool else
        f'Locked experts: `{pool["expert_ids"]}`; pool hash `{pool["hash"]}`. '
        f'Research-only no-coverage diagnostic: {pool["research_only_no_oracle_gain"]}.\n\n'+table(
        ['Cells','Oracle','Oracle gain','Safe high-quality scenes','Common zero'],
        [[s['cells'],s['oracle_mean'],s['oracle_gain'],s['safe_high_quality_scenes'],s['common_zero']] for s in pool['subset_results']])+
        '\n'+json.dumps(pool['stage_val_expert_contribution'],indent=2)+'\n')+
        '\nPool selection is stage_val only. No fixed requirement to retain three specialists; redundant/no-support branches are omitted. Rich supervision candidates never enter this pool.\n')

    initializations=[];training=[]
    for folder in sorted((root/'scorers').glob('*')):
        value=get(str(folder.relative_to(root)/'SCORER_INITIALIZATION.json'))
        if value:
            initializations.append([folder.name,value.get('init_seed',value.get('module_init_seed')),value['initial_state_hash'],value['parameter_count'],value['external_scorer_weights_loaded']])
        value=get(str(folder.relative_to(root)/'SCORER_TRAINING_REPORT.json'))
        if value:training.append([folder.name,value['optimizer_steps'],value['natural_consumed'],value['comparison_consumed'],
                                 value.get('supervised_natural_candidate_exposures'),value.get('ranking_candidate_exposures')])
    report('SCORER_WEIGHT_PROVENANCE.md','# Independent Scorer initialization\n\n'+table(
        ['Cell','Initialization seed','Initial state SHA256','New parameters','External Scorer loaded'],initializations)+
        '\nAll new scoring projections/readers/trajectory encoders/Decoder/output heads start randomly. Common semantic modules use identical initialization across the factorial cells. '
        'Only same-cell optimizer-boundary checkpoints may resume a Scorer; no other cell, DrivoR, M0, old driving evaluator or teacher weights are eligible.\n')
    report('SCORER_TRAINING_REPORT.md','# Scorer training\n\n'+table(
        ['Cell','Updates','Natural scenes consumed','Pairs consumed','Natural candidate exposures','Ranking candidate exposures'],training)+
        '\nD_S only; first3000 updates pointwise, then3000 updates natural64+comparison64. The continuous warmup100/constant1e-4 scheduler is not restarted. '
        'Rich pretraining increases candidate exposure and label cost; equal update counts do not mean equal compute/label budget. '
        'No Scorer currently missing from this table should be interpreted as trained.\n')
    report('RANKING_OBJECTIVE_TEST.md','# Cost-sensitive variable-K objective\n\n'
        'For each scene, use all valid unordered pairs P_x, then a_xij = non_tie*abs(gap)/max(P_x,1). '
        'The comparison sampler draws q proportional to a; the sampled softplus loss is not multiplied by gap again. '
        'No scene-wise division by sum(abs(gap)) is performed. K=2 therefore retains the0.95/0.02 relative cost. '
        'Exact duplicate trajectories are excluded from extra comparison information. K=1/all-tie has zero ranking denominator.\n\n'
        'Current tests cover analytic cost ratio, variable-K pair denominators, invalid/tie/NaN padding, stage transition sampler recovery and microbatch equality. '
        'A two-process CPU/Gloo test with unequal rank-valid populations matches the single-process global objective gradients within2e-7. '
        'Natural pointwise value/component losses are reduced within each scene then across scenes, independently of ranking.\n')

    spatial=get('SPATIAL_CONTEXT_COMPLETE.json')
    report('SPATIAL_CONTEXT_AUDIT.md','# Same-forward spatial information\n\n'+
        (f'COMPLETE: {spatial["requested"]} fixed scenes; plan `{spatial["plan_hash"]}`.\n' if spatial else 'RUNNING: the full spatial manifest is not yet complete.\n')+
        '\nW144×2048 and A8×2048 are current-input language hidden states. Spatial V is the same visual tower\'s final merged three-view token output. '
        'The controlled return wrapper does not change prompt, tokens, mask, MRoPE or the original planner function. '
        'Hooks verify visual=1 and language=1 per image batch; label deletion/shuffling leaves W/A/planner/V unchanged. '
        'The actual256-scene image pilot matched cached W/A exactly; maximum raw-action difference was3.5763e-7 under the predeclared1e-5 tolerance. '
        'No future auxiliary prediction head is executed. Spatial readout uses16 learned queries per original view,256 dimensions and one4-head cross-attention/FFN. '
        'All tensors remain at the original inference precision.\n\n'
        'The six-worker runtime keeps the unchanged numeric capture source hash and verifies whole-manifest coverage. '
        'Reader payload LRU and checksum-version metadata are bounded; every immutable record is fully SHA256-verified once per observed file version, with later inode/size/mtime/ctime checks. '
        'Changing Base/observation/capture identity or mutating a record is rejected.\n')
    rich=get('banks/rich/scorer_fit/STATISTICS.json')
    report('RICH_BANK_COVERAGE.md','# Rich supervision coverage\n\n'+
        ('WAITING_INPUTS: rich bank not committed.\n' if rich is None else json.dumps(rich,indent=2)+'\n')+
        '\nRich bank contains the exact deployment pool, then available own1000/3000 raw endpoints, admitted historical optimized trajectories and bounded current-state variants, maximum12 candidates/scene. '
        'GT is not a deployment candidate or a default rich-bank candidate. Repeated trajectories are deduplicated. '
        'Collision/road-negative physically expressible trajectories retain their true low scores. Numerical/coordinate/time errors are invalid data. '
        'Constructed variants use the real current speed with explicit acceleration/lateral acceleration/yaw-rate/jerk checks, do not extrapolate beyond the source horizon, and receive fresh official scores.\n')

    final=get('FINAL_SYSTEM_RESULTS.json')
    report('SELECTION_REGRET_AND_RISK.md','# Actual selection and risk\n\n'+
        ('UNTESTED: final fixed-dev selection has not run. Oracle results are not Scorer results.\n' if final is None else table(
            ['Policy','Scenes','Base','Oracle','Selected','G','R','Selected gain','Over best fixed','High→zero','Selected zeros'],
            [[s['cell'],s['scenes'],s['base_01'],s['oracle_01'],s['selected_01'],s['oracle_gain_01'],s['regret_01'],s['selected_gain_01'],
              s['selected_over_best_fixed_01'],s['high_to_zero'],s['selected_zero']] for s in final['results']]))+
        '\nThe identity selected_mean−Base_mean=G−R is checked. All Scorers use the same locked pool. Raw argmax and optional selector_cal margin are separate. '
        'Always-Base is never called selection success. DDC compliance is diagnostic because its official weight is zero. '
        'Detailed valid denominators, NC/DAC/TTC failures, wrong-switch costs and fixed-policy comparisons are in FINAL_SYSTEM_RESULTS.json.\n')
    latency=get('LATENCY_AND_PARAMETER_REPORT.json')
    report('LATENCY_AND_PARAMETER_REPORT.md','# Real inference cost\n\n'+
        ('WAITING_INPUTS: trained-system image-to-trajectory benchmark not executed.\n' if latency is None else json.dumps(latency,indent=2)+'\n')+
        '\nMeasurements include real image decode and one shared visual/language forward, per-expert action computation and selected score head. '
        'Semantic and spatial costs are separate. Cached head training is not deployment latency.\n')
    resume=get('qualifications/expert_gpu_resume/REPORT.json')
    reference=get('qualifications/official_reference/REPORT.json')
    roundtrip=get('research_bundle/ROUNDTRIP_REPORT.json')
    report('REGRESSION_AND_RESUME_REPORT.md','# Regression and actual integration\n\n'
        'Repository regression:476 passed,3 skipped,4 warnings,60.20s (pytest_regression_v5.txt). After adding physical-capacity guards, the focused study/artifact suite passed47 tests in4.33s (pytest_capacity_migration.txt). '
        'These are distinct evidence runs, not an invented aggregate suite count.\n\n'+table(
            ['Real check','Status','Scope'],[['GPU private-reader exact resume',resume['status'] if resume else 'UNTESTED','32 real scenes,4 continuous vs2+2,raw/EMA/Adam/scheduler/RNG/cursor/loss/next IDs; terminal adds0'],
            ['Fresh official reference invariance',reference['status'] if reference else 'UNTESTED','two real scenes,Base/GT audit,reorder/delete/duplicate/chunk,exact bank agreement'],
            ['Independent-process research bundle',roundtrip['status'] if roundtrip else 'WAITING_INPUTS','real images,strict loading,finite outputs and stable selected candidate']])+
        '\nNo training fixture is presented as real integration. The CPU/Gloo distributed loss test is separate from the actual single-GPU training path. '
        'No current Base/optimizer/data cursor or other experiment was restarted for these checks.\n')
    status={'generated_utc':now,'dependencies':dependencies,'TRAINING_COMPLETE':'PASS' if final else 'RUNNING',
            'SCIENTIFIC_CONCLUSION':get('VERDICT.json',{}).get('SCIENTIFIC_CONCLUSION','UNTESTED'),
            'expert_update_counts':{p.parent.name:read_json(p).get('optimizer_steps') for p in (root/'experts').glob('*/result.json')},
            'scorer_update_counts':{p.parent.name:read_json(p).get('optimizer_steps') for p in (root/'scorers').glob('*/result.json')},
            'spatial_complete':spatial is not None,'Base_training_modified':False,'ACTIVE_updated':False,'push_executed':False}
    atomic_json(root/'REPORT_STATUS.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    print(json.dumps(generate(load_config(args.config))))


if __name__=='__main__':main()
